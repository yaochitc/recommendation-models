package io.yaochi.recommendation.model.pnn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

class ProductEncoder(batchSize: Int,
                     nFields: Int,
                     embeddingDim: Int,
                     outputDim: Int,
                     mats: Array[Float],
                     start: Int = 0) {
  private val numPairs = nFields * (nFields - 1) / 2

  private val productOutLinearLayer = LayerUtil.buildLinear(nFields * embeddingDim, outputDim, mats, false, start)

  private val productOutModule = buildProductOutModule()

  private val productInnerLinearOffset = start + nFields * embeddingDim * outputDim

  private val productInnerLinearLayer = LayerUtil.buildLinear(numPairs, outputDim, mats, false, productInnerLinearOffset)

  private val productInnerModule = buildProductInnerModule()

  private val productOutputBiasOffset = productInnerLinearOffset + numPairs * outputDim

  private val productOutputBiasLayer = LayerUtil.buildBiasLayer(outputDim, mats, productOutputBiasOffset)

  private val productOutputModule = buildOutputModule()

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val productOutTensor = productOutModule.forward(input).toTensor[Float]
    val productInnerTensor = productInnerModule.forward(input).toTensor[Float]

    productOutputModule.forward(T.apply(productOutTensor, productInnerTensor)).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val productOutputInput = T.apply(
      productOutModule.output.toTensor[Float],
      productInnerModule.output.toTensor[Float]
    )

    val productOutputTable = productOutputModule.backward(productOutputInput, gradOutput).toTable
    val gradTensor = productOutModule.backward(input, productOutputTable[Tensor[Float]](1)).toTensor[Float]
      .add(productInnerModule.backward(input, productOutputTable[Tensor[Float]](2)).toTensor[Float])

    var curOffset = start
    BackwardUtil.linearBackward(productOutLinearLayer, mats, curOffset)

    curOffset = productInnerLinearOffset
    BackwardUtil.linearBackward(productInnerLinearLayer, mats, curOffset)

    curOffset = productOutputBiasOffset
    BackwardUtil.biasBackward(productOutputBiasLayer, mats, curOffset)

    gradTensor
  }

  def buildProductOutModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))
      .add(productOutLinearLayer)
  }

  def buildProductInnerModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields, embeddingDim), Some(false)))
      .add(InnerProduct())
      .add(productInnerLinearLayer)
  }

  def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(productOutputBiasLayer)
      .add(ReLU())
  }

  def getParameterSize: Int = {
    productInnerLinearOffset + numPairs * outputDim + 1
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
