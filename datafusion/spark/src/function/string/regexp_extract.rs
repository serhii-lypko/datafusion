use arrow::array::{
    Array, ArrayAccessor, ArrayBuilder, ArrayRef, AsArray, BinaryViewBuilder,
    GenericBinaryBuilder, GenericStringBuilder, Int64Array, OffsetSizeTrait,
    StringViewBuilder,
};

use arrow::datatypes::DataType;
use datafusion_common::{Result, exec_err, not_impl_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;

use regex::{CaptureLocations, Regex};

use std::collections::HashMap;
use std::sync::Arc;

/*
  ?NOTE: performance surface

  It's implicit. Scalar-pattern specialization, avoiding per-row compiles, Arrow builder choices
  (for example regexp_replace.rs's OptimizedRegex fast path and buffer-level construction).
*/

// TODO -> complete spark semantics
// - Spark null-intolerant semantics

/// Spark-compatible `regexp_extract` expression
/// <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
///
/// TODO -> describe semantics (like in substring)
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

// TODO -> what is ArrowBuilder choices? (GenericBinaryBuilder, GenericStringBuilder etc.)

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
        // ! 🟡 How exactly make_scalar_function re-collapses all-scalar inputs back to a Scalar at the end? 🟡

        // TODO variant -> use make_scalar_function + the HashMap cache. Clean, idiomatic, correct, matches substring.rs.
        // Add one doc line: "A constant pattern is compiled once via the cache; a dedicated Scalar
        // fast path (compile before the loop, skip expansion) is a possible further optimization."

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
    let pattern_array = args[1].as_string::<i32>();
    let idx_array = if args.len() > 2 {
        Some(as_int64_array(&args[2])?)
    } else {
        // Default as idx = 1
        None
    };

    // TODO -> figure out downcasting.

    match args[0].data_type() {
        // Utf8 is the Arrow type tag
        DataType::Utf8 => {
            let array = args[0].as_string::<i32>();

            // spark_substring_generic(
            //     &array,
            //     start_array,
            //     length_array,
            //     GenericStringBuilder::<i32>::new(),
            //     is_ascii,
            // )
        }
        DataType::LargeUtf8 => {
            let array = args[0].as_string::<i64>();

            // spark_substring_generic(
            //     &array,
            //     start_array,
            //     length_array,
            //     GenericStringBuilder::<i64>::new(),
            //     is_ascii,
            // )
        }
        DataType::Utf8View => {
            let array = args[0].as_string_view();

            // spark_substring_generic(
            //     &array,
            //     start_array,
            //     length_array,
            //     StringViewBuilder::new(),
            //     is_ascii,
            // )
        }
        other => exec_err!(
            "Unsupported data type {other:?} for function spark_regexp_extract, expected Utf8View, Utf8, LargeUtf8."
        ),
    }

    // creating Regex is expensive so create hashmap for memoization
    // must have cache - never compile per-row
    // one map per batch
    let mut patterns: HashMap<String, Regex> = HashMap::new();

    /*
      Reuse from regexp_replace (shape only): the HashMap<String,Regex> cache, the ArrayIter+zip per-row loop,
      the match datatype { Utf8/LargeUtf8 => ..., Utf8View => ... } collect, and (if you add flags) the (?flags) prepend trick.
    */

    todo!()
}

// TODO -> consider inlining some of the methods/handlers?

/*
  Reasoning notes

  - Why not PCRE (backtracking engine).
  A vectorized engine processing untrusted user patterns cannot accept exponential backtracking —
  linear-time guarantee is a feature Spark itself lacks (Java regex can blow up).
  RE2 is O(nm) - input x pattern_size
  TODO -> define overview what gets missed when choosing RE2 over PCRE. And what are benefits.

  - RegExp engines mismatch.
  Patterns using backreferences/lookaround succeed in Spark and error in given impl.
  Everything else matches.

   - Why not just wrap datafusion/functions/src/regex/regexpreplace.rs
  Wrong operation: it replaces matched text and returns the modified string. Wrong signature. Wrong output rules.

  - Escaping.
  TODO

  - Adding/skipping flags (see regexp_replace).
  TODO -> most reasonable would be mention them, but point as skipped in terms of time spent

  - Micro opmimization: hand-roll alternative to make_scalar_function
  The legit reason you might hand-roll it: compile a constant pattern exactly once. With make_scalar_function,
  a constant pattern gets expanded to N identical strings, and your HashMap<String,Regex> cache collapses them
  back to one compile — but you pay N cache lookups + the array materialization. Hand-rolling lets you check
  "is the pattern a Scalar? compile once, skip the map."

*/

#[cfg(test)]
mod tests {
    use super::*;

    // TODO -> signature match/mismatch
}
