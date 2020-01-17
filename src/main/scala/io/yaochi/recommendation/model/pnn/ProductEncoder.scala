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

  private val productOutLinearLayer = LayerUtil.buildLinear(nFields * embeddingDim, outputDim, mats, false, start)

  private val productOutModule = buildProductOutModule()

  private val productInnerLinearOffset = start + nFields * embeddingDim * outputDim

  private val productInnerLinearLayer = LayerUtil.buildLinear(numPairs, outputDim, mats, false, productInnerLinearOffset)

  private val productInnerReshape = Reshape[Float](Array(batchSize, nFields, embeddingDim), Some(false))

  private val productInnerModule = buildProductInnerModule()

  private val productOutputBiasOffset = productInnerLinearOffset + numPairs * outputDim

  private val productOutputBiasLayer = LayerUtil.buildBiasLayer(outputDim, mats, productOutputBiasOffset)

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

  def buildProductInnerModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Gather(batchSize, numPairs, embeddingDim))
      .add(DotProduct2())
      .add(productInnerLinearLayer)
  }

  def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(productOutputBiasLayer)
      .add(ReLU())
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
