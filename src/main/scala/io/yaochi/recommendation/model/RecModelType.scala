package io.yaochi.recommendation.model

object RecModelType extends Enumeration {
  type RecModelType = Value
  val BIAS_WEIGHT = Value("BIAS_WEIGHT")
  val BIAS_WEIGHT_EMBEDDING = Value("BIAS_WEIGHT_EMBEDDING")
  val BIAS_WEIGHT_EMBEDDING_MATS = Value("BIAS_WEIGHT_EMBEDDING_MATS")
  val BIAS_WEIGHT_EMBEDDING_MATS_FIELD = Value("BIAS_WEIGHT_EMBEDDING_MATS_FIELD")
}
