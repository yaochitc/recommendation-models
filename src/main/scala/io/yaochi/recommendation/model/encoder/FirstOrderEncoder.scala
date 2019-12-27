package io.yaochi.recommendation.model.encoder

import com.intel.analytics.bigdl.nn.Scatter
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

class FirstOrderEncoder(batchSize: Int) {
  private val module = new Scatter[Float](batchSize, 1)

  def forward(input: Table): Tensor[Float] = {
    module.forward(input)
  }

  def backward(input: Table, gradOutput: Tensor[Float]): Table = {
    module.backward(input, gradOutput)
  }
}

object FirstOrderEncoder {
  def apply(batchSize: Int): FirstOrderEncoder = new FirstOrderEncoder(batchSize)
}
