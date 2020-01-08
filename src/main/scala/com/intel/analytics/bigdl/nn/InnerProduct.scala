package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class InnerProduct[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val (_, batch, len) = getShape(input)
    output.resize(batch, len * (len - 1) / 2)

    var cc = 1
    var i = 1
    var j = 2
    while (i < len) {
      val ijDot = batchDot(input.select(2, i), input.select(2, j))
      output.select(2, cc).copy(ijDot)

      cc += 1
      if (j == len) {
        i += 1
        j = i + 1
      } else {
        j += 1
      }
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }

    val gout = gradOutput

    require(gout.dim() == 2, s"invalid dim of gradOutput(${gout.dim()})!")

    val (dim, _, len) = getShape(input)

    val outLen = len * (len - 1) / 2
    require(gout.size(2) == outLen,
      s"invalid colSize of gradOutput(${gout.size(2)}), it should be $outLen!")

    val emLen = getEmbeddingSize(input)

    var cc = 1
    var i = 1
    var j = 2
    while (i < len) {
      val (ti, tj) = dim match {
        case 2 =>
          input.select(2, i).view(1, emLen) -> input.select(2, j).view(1, emLen)
        case 3 =>
          input.select(2, i) -> input.select(2, j)
      }

      // get cc_th column data from total gradOut
      val go = gout.narrow(2, cc, 1)

      val jInc = Tensor[T]().resizeAs(ti).copy(ti).cmul(go)
      if (dim == 2) jInc.squeeze()
      gradInput.select(2, j).add(jInc)

      val iInc = Tensor[T]().resizeAs(tj).copy(tj).cmul(go)
      if (dim == 2) iInc.squeeze()
      gradInput.select(2, i).add(iInc)

      cc += 1
      if (j == len) {
        i += 1
        j = i + 1
      } else {
        j += 1
      }
    }

    gradInput
  }

  private def getEmbeddingSize(t: Tensor[T]): Int = {
    if (t.dim() == 2) t.size(2) else t.size(3)
  }

  private def batchDot(t1: Tensor[T], t2: Tensor[T]): Tensor[T] = {
    var (input1, input2) = (t1, t2)

    if (input1.dim() == 1) {
      input1 = input1.view(1, input1.size(1))
      input2 = input2.view(1, input2.size(1))
    }

    val buffer = Tensor[T]()
    buffer.resizeAs(input1).cmul(input1, input2)
    buffer.sum(2).squeeze()
  }

  private def getShape(t: Tensor[T]) = {
    val (batch, size) = t.dim() match {
      case 2 => 1 -> t.size(1)
      case 3 => t.size(1) -> t.size(2)
      case n => throw new IllegalArgumentException(s"wrong dim of input Tensor($n)!")
    }
    (t.dim(), batch, size)
  }

}

object InnerProduct {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): InnerProduct[T] = new InnerProduct()
}