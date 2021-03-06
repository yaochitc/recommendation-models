package io.yaochi.recommendation.optim

import java.util.concurrent.Future

import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.ml.matrix.psf.update.base.{UpdateParam, VoidResult}
import com.tencent.angel.spark.ml.psf.optim.{AsyncMomentumFunc, AsyncOptimParam}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}

class AsyncMomentum(eta: Double, momentum: Double = 0.9) extends AsyncOptim {

  override def getNumSlots(): Int = 2

  def getParam(matrixId: Int, grads: Array[Vector], offset: Int): UpdateParam =
    new AsyncOptimParam(matrixId, grads, Array(eta, momentum), Array(offset, getNumSlots()))

  override def asycUpdate(vector: PSVector, offset: Int, grad: Vector): Future[VoidResult] = {
    grad.setRowId(vector.id)
    val func = new AsyncMomentumFunc(getParam(vector.poolId, Array(grad), offset))
    vector.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, rowIds: Array[Int], grads: Array[Vector]): Future[VoidResult] = {
    assert(grads.length == rowIds.length)
    grads.zip(rowIds).map(f => f._1.setRowId(f._2))
    val func = new AsyncMomentumFunc(getParam(matrix.id, grads, offset))
    matrix.psfUpdate(func)
  }

  override def asycUpdate(matrix: PSMatrix, offset: Int, grads: Array[Vector]): Future[VoidResult] = {
    asycUpdate(matrix, offset, (0 until grads.length).toArray, grads)
  }
}
