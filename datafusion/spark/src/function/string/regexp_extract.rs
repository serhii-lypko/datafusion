// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{
    Array, ArrayAccessor, ArrayBuilder, ArrayIter, ArrayRef, AsArray,
    GenericStringBuilder, Int64Array, OffsetSizeTrait, StringViewBuilder,
};
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int64_array;
use datafusion_common::{DataFusionError, Result, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use regex::Regex;
use std::collections::HashMap;

/*
  ?NOTE: performance surface

  It's implicit. Scalar-pattern specialization, avoiding per-row compiles, Arrow builder choices
  (for example regexp_replace.rs's OptimizedRegex fast path and buffer-level construction).
*/

// TODO -> don't forget to implement Catalyst's RegExpExtract 2-arg form
// ->> one UDF, 2-or-3 arity, column-capable
// ->> what's left: switch idx from Exact(Int64) to Coercible so Int32 also matches (Spark's idx is Int32).

/// Spark-compatible `regexp_extract` expression.
///
/// `regexp_extract(str, regexp[, idx])` - extracts the first match of `regexp`
/// in `str` and returns capture group `idx` (default 1; 0 = whole match).
///
/// Docs:
/// - <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
/// - <https://docs.databricks.com/aws/en/sql/language-manual/functions/regexp_extract>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        // Spark: regexp_extract(str, regexp[, idx]); idx defaults to 1.
        use DataType::*;
        Self {
            // accepts only concrete types, no conversion. If a variant doesn't match exactly, next one will be tried.
            signature: Signature::one_of(
                vec![
                    // (str, regexp, idx)
                    TypeSignature::Exact(vec![Utf8View, Utf8View, Int64]),
                    TypeSignature::Exact(vec![Utf8, Utf8, Int64]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8, Int64]),
                    // (str, regexp) -> idx defaults to 1
                    TypeSignature::Exact(vec![Utf8View, Utf8View]),
                    TypeSignature::Exact(vec![Utf8, Utf8]),
                    TypeSignature::Exact(vec![LargeUtf8, LargeUtf8]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkRegexpExtract {
    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    // TODO -> need to figure out.
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Spark: output type == subject (arg 0) type.
        // TODO -> null semantics via return_field_from_args (null if any input null).
        Ok(arg_types[0].clone())
    }

    // TODO -> implement?
    // fn return_field_from_args() {}

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_regexp_extract, vec![])(&args.args)
    }
}

// From the datafusion/functions/src/regex/regexpreplace.rs
// the array-plumbing + cache skeleton transfers

/*
  *Scalars caching example

  SELECT regexp_extract('100-200', '(\\d+)-(\\d+)', 1);

  - '100-200' → Scalar(Utf8("100-200"))
  - '(\d+)-(\d+)' → Scalar(Utf8("(\d+)-(\d+)")) (the SQL \\d unescapes to \d)
  - 1 → Scalar(Int32(1))

  >> In the cache: only the pattern. One entry:
  key:   "(\d+)-(\d+)"        // the pattern string
  value: Regex(/(\d+)-(\d+)/)  // compiled

  The subject and idx are not cached — the cache exists only to avoid recompiling regexes,
  and neither '100-200' nor 1 gets compiled. No reason to store them:
  subject and idx just get expanded (copied) to arrays and read per-row — no caching,
  because reading a value is free. Can only cache things that are expensive to produce,
  and a regex is the only expensive thing here (compilation). A string or an int is
  cheap to just copy and read.
*/

fn spark_regexp_extract(args: &[ArrayRef]) -> Result<ArrayRef> {
    // idx (arg 2) is optional; Spark defaults it to 1.
    let idx_array = if args.len() > 2 {
        Some(as_int64_array(&args[2])?)
    } else {
        None
    };

    // Dispatch on the subject's string type. The signature guarantees `str` and
    // `regexp` share the same string type, so both downcast the same way. Output
    // mirrors the input string type. Binary / other -> error.
    match args[0].data_type() {
        DataType::Utf8 => spark_regexp_generic(
            args[0].as_string::<i32>(),
            args[1].as_string::<i32>(),
            idx_array,
            GenericStringBuilder::<i32>::new(),
        ),
        DataType::LargeUtf8 => spark_regexp_generic(
            args[0].as_string::<i64>(),
            args[1].as_string::<i64>(),
            idx_array,
            GenericStringBuilder::<i64>::new(),
        ),
        DataType::Utf8View => spark_regexp_generic(
            args[0].as_string_view(),
            args[1].as_string_view(),
            idx_array,
            StringViewBuilder::new(),
        ),
        other => exec_err!(
            "Unsupported data type {other:?} for function regexp_extract, expected Utf8, LargeUtf8 or Utf8View."
        ),
    }
}

trait RegexpExtractBuilder: ArrayBuilder {
    fn append_value(&mut self, val: &str);
    fn append_null(&mut self);
}

impl<O: OffsetSizeTrait> RegexpExtractBuilder for GenericStringBuilder<O> {
    fn append_value(&mut self, val: &str) {
        GenericStringBuilder::append_value(self, val);
    }
    fn append_null(&mut self) {
        GenericStringBuilder::append_null(self);
    }
}

impl RegexpExtractBuilder for StringViewBuilder {
    fn append_value(&mut self, val: &str) {
        StringViewBuilder::append_value(self, val);
    }
    fn append_null(&mut self) {
        StringViewBuilder::append_null(self);
    }
}

/// Batch-level driver. Holds all Spark `RegExpExtractBase.extract` semantics:
///   - null-intolerant: any null input (subject/pattern/idx) -> null row
///   - no match -> "" (empty string, NOT null)
///   - idx validated only AFTER a match: idx < 0 || idx > group_count -> error
///     (so a no-match with an out-of-range idx still returns "")
///   - optional group that did not participate -> ""
fn spark_regexp_generic<'a, S, Builder>(
    subject: S,
    pattern: S,
    idx_array: Option<&Int64Array>,
    mut builder: Builder,
) -> Result<ArrayRef>
where
    S: ArrayAccessor<Item = &'a str>,
    Builder: RegexpExtractBuilder,
{
    let mut patterns: HashMap<String, Regex> = HashMap::new();

    for (row, (s, p)) in ArrayIter::new(subject)
        .zip(ArrayIter::new(pattern))
        .enumerate()
    {
        // idx defaults to 1 (Catalyst 2-arg form); a null idx nulls the row.
        let idx = match idx_array {
            Some(a) if a.is_null(row) => None,
            Some(a) => Some(a.value(row)),
            None => Some(1_i64),
        };

        // Spark: null-intolerant (any null input -> null output).
        let (s, p, idx) = match (s, p, idx) {
            (Some(s), Some(p), Some(idx)) => (s, p, idx),
            _ => {
                builder.append_null();
                continue;
            }
        };

        if !patterns.contains_key(p) {
            // RE2 rejects PCRE-only constructs (backreferences, lookaround).
            let re = Regex::new(p).map_err(|e| {
                DataFusionError::Execution(format!(
                    "regexp_extract: failed to compile pattern '{p}': {e}"
                ))
            })?;
            patterns.insert(p.to_string(), re);
        }
        let re = &patterns[p];

        match re.captures(s) {
            None => builder.append_value(""),
            Some(caps) => {
                let group_count = caps.len() - 1;
                if idx < 0 || idx as usize > group_count {
                    return exec_err!(
                        "regexp_extract: invalid group index {idx}, pattern has {group_count} group(s)"
                    );
                }

                // Optional group that did not participate -> "".
                builder.append_value(caps.get(idx as usize).map_or("", |m| m.as_str()));
            }
        }
    }

    Ok(builder.finish())
}

// TODO -> consider inlining some of the methods/handlers?

/*
  --->> Design notes <<---

  - Why not PCRE (backtracking engine).
  A vectorized engine processing untrusted user patterns cannot accept exponential backtracking —
  linear-time guarantee is a feature Spark itself lacks (Java regex can blow up).
  RE2 is O(nm) - input x pattern_size
  TODO -> define overview what gets missed when choosing RE2 over PCRE. And what are benefits.

  - RegExp engines mismatch.
  Patterns using backreferences/lookaround succeed in Spark and error in given impl.
  Everything else matches.

  - Using make_scalar_function.
  Micro opmimization vector: hand-roll alternative to make_scalar_function.
  The legit reason you might hand-roll it: compile a constant pattern exactly once. With make_scalar_function,
  a constant pattern gets expanded to N identical strings, and your HashMap<String,Regex> cache collapses them
  back to one compile — but you pay N cache lookups + the array materialization. Hand-rolling lets you check
  "is the pattern a Scalar? compile once, skip the map."

  - Using builder over Vec<Option<String>>
  TODO: more descriptive.
  Clean design fit. Moderate performance optimization.
  One heap allocation per row + the Vec. The final copy into the output buffer happens either way.

   - Why not just wrap datafusion/functions/src/regex/regexpreplace.rs
  Wrong operation: it replaces matched text and returns the modified string. Wrong signature. Wrong output rules.

  - Escaping.
  TODO

  - Adding/skipping flags (see regexp_replace).
  TODO -> most reasonable would be mention them, but point as skipped in terms of time spent

  ...

  --->> Further performance vectors <<---

  - Arrow builder
  regexp_replace.rs's OptimizedRegex fast path (rewrites anchored single-group patterns for direct extraction)
  and buffer-level output construction (write values/offsets buffers directly, skipping the builder).
  Both are replace-specific micro-opts; correctness-focused pass uses the pattern cache + standard string builders instead.

*/

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    // Focused on the Spark-semantics corner cases.

    /// Single-row helper: drives the real batch entry point.
    fn run_one(pattern: &str, subject: &str, idx: i64) -> Result<String> {
        let s = StringArray::from(vec![subject]);
        let p = StringArray::from(vec![pattern]);
        let i = Int64Array::from(vec![idx]);
        let out =
            spark_regexp_generic(&s, &p, Some(&i), GenericStringBuilder::<i32>::new())?;
        Ok(out.as_string::<i32>().value(0).to_string())
    }

    #[test]
    fn basic_group_extraction() {
        assert_eq!(run_one(r"(\d+)-(\d+)", "100-200", 1).unwrap(), "100");
        assert_eq!(run_one(r"(\d+)-(\d+)", "100-200", 2).unwrap(), "200");
    }

    #[test]
    fn idx_zero_is_whole_match() {
        assert_eq!(run_one(r"(\d+)-(\d+)", "100-200", 0).unwrap(), "100-200");
    }

    #[test]
    fn no_match_returns_empty_not_null() {
        // Invariant: no match -> "" (never null).
        assert_eq!(run_one(r"(\d+)", "abc", 1).unwrap(), "");
    }

    #[test]
    fn optional_group_absent_returns_empty() {
        // "(a)(x)?" matches "a"; group 2 is optional and absent -> "".
        assert_eq!(run_one(r"(a)(x)?", "abc", 2).unwrap(), "");
    }

    #[test]
    fn idx_out_of_range_errors() {
        let err = run_one(r"(\d+)-(\d+)", "100-200", 5).unwrap_err();
        assert!(err.to_string().contains("invalid group index"));
    }

    #[test]
    fn negative_idx_with_match_errors() {
        let err = run_one(r"(a)", "abc", -1).unwrap_err();
        assert!(err.to_string().contains("invalid group index"));
    }

    #[test]
    fn out_of_range_idx_with_no_match_returns_empty() {
        // Nuance: idx is checked only after a match. No match -> "" (no error).
        assert_eq!(run_one(r"(\d+)", "abc", 9).unwrap(), "");
    }

    #[test]
    fn unsupported_pattern_errors() {
        // RE2 has no backreferences; Spark (Java) would accept this.
        let err = run_one(r"(a)\1", "aa", 1).unwrap_err();
        assert!(err.to_string().contains("failed to compile pattern"));
    }

    #[test]
    fn null_input_yields_null_row() {
        // Array level: null subject -> null output (null-intolerant).
        let subject = StringArray::from(vec![Some("100-200"), None]);
        let pattern = StringArray::from(vec![Some(r"(\d+)"), Some(r"(\d+)")]);
        let out = spark_regexp_generic(
            &subject,
            &pattern,
            None,
            GenericStringBuilder::<i32>::new(),
        )
        .unwrap();
        let out = out.as_string::<i32>();
        assert_eq!(out.value(0), "100");
        assert!(out.is_null(1));
    }
}
