package io.yaochi.recommendation.model

import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import io.yaochi.recommendation.model.RecModelType.RecModelType

abstract class RecModel(val `type`: RecModelType) extends Serializable {
  def getType: RecModelType = `type`

  def forward(batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch)
    forward(params)
  }

  def forward(batchSize: Int, batch: CooLongFloatMatrix, fields: Array[Long]): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch) ++ Map(
      "fields" -> fields
    )
    forward(params)
  }

  def forward(batchSize: Int, batch: CooLongFloatMatrix,
              bias: Array[Float], weights: Array[Float]): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights)
    forward(params)
  }

  def forward(batchSize: Int, batch: CooLongFloatMatrix,
              bias: Array[Float], weights: Array[Float],
              embeddings: Array[Float], embeddingDim: Int): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim
    )
    forward(params)
  }

  def forward(batchSize: Int, batch: CooLongFloatMatrix,
              bias: Array[Float], weights: Array[Float],
              embeddings: Array[Float], embeddingDim: Int,
              mats: Array[Float], matSizes: Array[Int]): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim,
      "mats" -> mats,
      "mats_sizes" -> matSizes
    )
    forward(params)
  }

  def forward(batchSize: Int, batch: CooLongFloatMatrix,
              bias: Array[Float], weights: Array[Float],
              embeddings: Array[Float], embeddingDim: Int,
              mats: Array[Float], matSizes: Array[Int],
              fields: Array[Long]): Array[Float] = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim,
      "mats" -> mats,
      "mats_sizes" -> matSizes,
      "fields" -> fields
    )
    forward(params)
  }

  def backward(batchSize: Int, batch: CooLongFloatMatrix,
               bias: Array[Float], weights: Array[Float],
               targets: Array[Float]): Float = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "targets" -> targets
    )
    backward(params)
  }

  def backward(batchSize: Int, batch: CooLongFloatMatrix,
               bias: Array[Float], weights: Array[Float],
               embeddings: Array[Float], embeddingDim: Int,
               targets: Array[Float]): Float = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim,
      "targets" -> targets
    )
    backward(params)
  }

  def backward(batchSize: Int, batch: CooLongFloatMatrix,
               bias: Array[Float], weights: Array[Float],
               embeddings: Array[Float], embeddingDim: Int,
               mats: Array[Float], matSizes: Array[Int],
               targets: Array[Float]): Float = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim,
      "mats" -> mats,
      "mats_sizes" -> matSizes,
      "targets" -> targets
    )
    backward(params)
  }

  def backward(batchSize: Int, batch: CooLongFloatMatrix,
               bias: Array[Float], weights: Array[Float],
               embeddings: Array[Float], embeddingDim: Int,
               mats: Array[Float], matSizes: Array[Int],
               fields: Array[Long], targets: Array[Float]): Float = {
    val params = RecModel.buildParams(batchSize, batch, bias, weights) ++ Map(
      "embedding" -> embeddings,
      "embedding_dim" -> embeddingDim,
      "mats" -> mats,
      "mats_sizes" -> matSizes,
      "fields" -> fields,
      "targets" -> targets
    )
    backward(params)
  }

  protected def forward(params: Map[String, Any]): Array[Float]

  protected def backward(params: Map[String, Any]): Float

  def getMatsSize: Array[Int]

  def getInputDim: Int

  def getEmbeddingDim: Int

}

object RecModel {
  def buildParams(batchSize: Int, batch: CooLongFloatMatrix): Map[String, Any] = {
    Map(
      "batch_size" -> batchSize,
      "index" -> batch.getRowIndices,
      "feats" -> batch.getColIndices,
      "values" -> batch.getRowIndices
    )
  }

  def buildParams(bias: Array[Float], weights: Array[Float]): Map[String, Any] = {
    Map(
      "bias" -> bias,
      "weights" -> weights
    )
  }

  def buildParams(batchSize: Int, batch: CooLongFloatMatrix, bias: Array[Float], weights: Array[Float]): Map[String, Any] = {
    Map(
      "batch_size" -> batchSize,
      "index" -> batch.getRowIndices,
      "feats" -> batch.getColIndices,
      "values" -> batch.getRowIndices,
      "bias" -> bias,
      "weights" -> weights
    )
  }
}