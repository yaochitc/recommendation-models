package io.yaochi.recommendation.model.pnn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

import scala.collection.mutable.ArrayBuffer

class ProductEncoder(batchSize: Int,
                     nFields: Int,
                     embeddingDim: Int,
                     outputDim: Int,
                     mats: Array[Float],
                     start: Int = 0) {
  private val numPairs = nFields * (nFields - 1) / 2

  private val (rowTensor, colTensor) = calcIndices()

  private val (productOutLinearLayer, productOutLinearEndOffset) = buildProductOutLinearLayer()

  private val productOutModule = buildProductOutModule()

  private val (productInnerLinearLayer, productInnerLinearEndOffset) = buildProductInnerLinearLayer()

  private val productInnerReshape = Reshape[Float](Array(batchSize, nFields, embeddingDim), Some(false))

  private val productInnerModule = buildProductInnerModule()

  private val (productOutputBiasLayer, productOutputBiasEndOffset) = buildProductOutputBiasLayer()

  private val productOutputModule = buildOutputModule()

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val productOutTensor = productOutModule.forward(input).toTensor[Float]
    val productInnerReshapeTensor = productInnerReshape.forward(input)
    val productInnerTensor = productInnerModule.forward(T.apply(productInnerReshapeTensor, rowTensor, colTensor))
      .toTensor[Float]

    productOutputModule.forward(T.apply(productOutTensor, productInnerTensor)).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val productOutputInput = T.apply(
      productOutModule.output.toTensor[Float],
      productInnerModule.output.toTensor[Float]
    )

    val productOutputGradTable = productOutputModule.backward(productOutputInput, gradOutput).toTable
    val gradTensor = productOutModule.backward(input, productOutputGradTable[Tensor[Float]](1)).toTensor[Float]

    val productInnerInput = T.apply(
      productInnerReshape.output.toTensor[Float],
      rowTensor,
      colTensor
    )
    val productInnerGradTable = productInnerModule.backward(productInnerInput, productOutputGradTable[Tensor[Float]](2))
      .toTable

    gradTensor.add(productInnerReshape.backward(input, productInnerGradTable[Tensor[Float]](1)).toTensor[Float])

    var curOffset = start
    curOffset = BackwardUtil.linearBackward(productOutLinearLayer, mats, curOffset)

    curOffset = BackwardUtil.linearBackward(productInnerLinearLayer, mats, curOffset)

    BackwardUtil.biasBackward(productOutputBiasLayer, mats, curOffset)

    gradTensor
  }

  def buildProductOutModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))
      .add(productOutLinearLayer)
  }

  def buildProductOutLinearLayer(): (Linear[Float], Int) = {
    val layer = LayerUtil.buildLinear(nFields * embeddingDim, outputDim, mats, false, start)
    val curOffset = start + nFields * embeddingDim * outputDim
    (layer, curOffset)
  }

  def buildProductInnerModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Gather(batchSize, numPairs, embeddingDim))
      .add(DotProduct2())
      .add(productInnerLinearLayer)
  }

  def buildProductInnerLinearLayer(): (Linear[Float], Int) = {
    val layer = LayerUtil.buildLinear(numPairs, outputDim, mats, false, productOutLinearEndOffset)
    val curOffset = productOutLinearEndOffset + numPairs * outputDim
    (layer, curOffset)
  }

  def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(productOutputBiasLayer)
      .add(ReLU())
  }

  def buildProductOutputBiasLayer(): (CAdd[Float], Int) = {
    val layer = LayerUtil.buildBiasLayer(1, mats, productInnerLinearEndOffset)
    val curOffset = productInnerLinearEndOffset + 1
    (layer, curOffset)
  }

  def calcIndices(): (Tensor[Int], Tensor[Int]) = {
    val rows = ArrayBuffer[Int]()
    val cols = ArrayBuffer[Int]()
    for (i <- 0 until nFields; j <- i + 1 until nFields) {
      rows += i
      cols += j
    }

    (Tensor.apply(rows.toArray, Array(rows.length)),
      Tensor.apply(cols.toArray, Array(cols.length)))
  }

  def getEndOffset: Int = {
    productOutputBiasEndOffset
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
