package io.yaochi.recommendation.model.pnn

import com.intel.analytics.bigdl.nn.{Linear, ReLU, Reshape, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import io.yaochi.recommendation.util.LayerUtil

class ProductEncoder(batchSize: Int,
                     nFields: Int,
                     embeddingDim: Int,
                     outputSize: Int,
                     mats: Array[Float],
                     start: Int = 0) {
  private val numPairs = nFields * (nFields - 1) / 2

  private val productOutLinearLayer = buildProductOutLinearLayer()

  private val productOutModule = buildProductOutModule()

  private val productInnerLinearOffset = start + nFields * embeddingDim * outputSize

  private val productInnerLinearLayer = buildProductInnerLinearLayer()

  def forward(input: Tensor[Float]): Tensor[Float] = {
    productOutModule.forward(input).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    productOutModule.backward(input, gradOutput).toTensor[Float]
  }

  def buildProductOutModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))
      .add(productOutLinearLayer)
      .add(ReLU())
  }

  def buildProductOutLinearLayer(): Linear[Float] = {
    LayerUtil.buildLinear(nFields * embeddingDim, outputSize, mats, false, start)
  }

  def buildProductInnerLinearLayer(): Linear[Float] = {
    LayerUtil.buildLinear(numPairs, outputSize, mats, false, productInnerLinearOffset)
  }

  def getParameterSize: Int = {
    productInnerLinearOffset + numPairs * outputSize
  }
}

object ProductEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            outputSize: Int,
            mats: Array[Float],
            start: Int = 0): ProductEncoder = new ProductEncoder(batchSize, nFields, embeddingDim, outputSize, mats, start)
}
