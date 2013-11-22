package cc.factorie.util

import java.util.concurrent.{Executors, Callable, ExecutorService}
import akka.actor.{Actor, Props, ActorSystem}
import concurrent.duration.Duration
import concurrent.{Future, Await}
import akka.actor.Status.Success

/**
 * User: apassos
 * Date: 7/30/13
 * Time: 2:57 PM
 */
object Threading {
  import scala.collection.JavaConversions._
  def parForeach[In](xs: Iterable[In], numThreads: Int)(body: In => Unit): Unit = withThreadPool(numThreads)(p => parForeach(xs, p)(body))
  def parForeach[In](xs: Iterable[In], pool: ExecutorService)(body: In => Unit): Unit = {
    val futures = xs.map(x => javaAction(body(x)))
    pool.invokeAll(futures).toSeq
  }

  def parMap[In, Out](xs: Iterable[In], numThreads: Int)(body: In => Out): Iterable[Out] = withThreadPool(numThreads)(p => parMap(xs, p)(body))
  def parMap[In, Out](xs: Iterable[In], pool: ExecutorService)(body: In => Out): Iterable[Out] = {
    val futures = xs.map(x => javaClosure(body(x)))
    pool.invokeAll(futures).toSeq.map(_.get())
  }

  def javaAction(in: => Unit): Callable[Object] = new Callable[Object] { def call(): Object = {
    try {
      in
    } catch {
      case t: Throwable => t.printStackTrace(); throw new Error("Caught error in parallel computation", t)
    }
    null
  }}
  def javaClosure[A](in: => A): Callable[A] = new Callable[A] { def call(): A =  try { in } catch { case t: Throwable => t.printStackTrace(); throw new Error(t) } }

  def newFixedThreadPool(numThreads: Int) = Executors.newFixedThreadPool(numThreads)
  def withThreadPool[A](numThreads: Int)(body: ExecutorService => A) = {
    val pool = newFixedThreadPool(numThreads)
    try {
      body(pool)
    } finally pool.shutdown()
  }
}

object ProducerConsumerProcessing{
  import akka.pattern.ask

  object IteratorMutex
  def parForeach[In](xs: Iterator[In], numParallelJobs: Int = Runtime.getRuntime.availableProcessors(),perJobTimeout: Long = 10 ,overallTimeout: Long = 24)(body: In => Unit): Unit  = {
    val system = ActorSystem("producer-consumer")

    val actors = (0 until numParallelJobs).map(i => system.actorOf(Props(new ParForeachActor(body)), "actor-"+i))


    val futures = actors.map(a => a.ask(Message(xs))(Duration(perJobTimeout,"minutes")))
    Await.result(Future.sequence(futures.toSeq), Duration(overallTimeout,"hours"))
    system.shutdown()

  }

  class ParForeachActor[In](function: In => Unit) extends Actor {
    def getNext(a: Iterator[In]): Option[In] = {
      IteratorMutex synchronized {
        if (a.hasNext)
          Some(a.next())
        else
          None
      }
    }
    def receive = {
      case Message(a) => {
        var stillWorking = true
        while (stillWorking) {
          val next = getNext(a.asInstanceOf[Iterator[In]])
          if(next.isDefined) function(next.get) else stillWorking = false
        }
        sender ! Success
      }
    }
  }
  case class Message[T](a: Iterator[T])

}

