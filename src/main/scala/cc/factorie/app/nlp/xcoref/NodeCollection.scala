package cc.factorie.app.nlp.xcoref

import cc.factorie.db.mongo.{LazyCubbieConverter, MongoCubbieCollection, MutableCubbieCollection}
import cc.factorie._
import cc.factorie.db.mongo.MongoCubbieImplicits._
import scala.collection.mutable
import com.mongodb.DB
import cc.factorie.app.nlp.hcoref.BagOfWordsVariable
import scala.reflect.ClassTag

/**
 * User: escher
 * Date: 11/2/13
 */
trait NodeCollection[Vars <: NodeVariables[Vars], N <: Node[Vars]] {
  def +=(n :N)                       :Unit
  def ++=(es:Iterable[N])            :Unit
  def drop()                         :Unit
  def store(nodesToStore:Iterable[N]):Unit
  def nextBatch(n:Int=10)            :Seq[N]
  def loadAll                        :Seq[N]
  //  def loadByIds(ids:Seq[String])     :Seq[N]
}

class BOWCubbie extends Cubbie{
  val nodeId = RefSlot[NodeCubbie[_,_]]("nid",()=>null.asInstanceOf[NodeCubbie[_,_]])  //todo understand how this null works
  val word   = StringSlot("w")
  val count  = DoubleSlot("c")
  def fetch = this.word -> this.count
  def store(id:String, w:String, c:Double) = {
    nodeId := id
    word   := w
    count  := c
    this
  }
}

trait DBNodeCollection[Vars <: NodeVariables[Vars], N <: Node[Vars] with Persistence, NC<:NodeCubbie[Vars, N]] extends NodeCollection[Vars, N]{
  protected val _id2cubbie = mutable.HashMap[String,NC]()
  protected def newNodeCubbie :NC
  protected def newNode(v:Vars, nc:NC) :N
  protected def cubbify(n:N) = {val nc = newNodeCubbie; nc.store(n); nc}
  protected def nodeCubbieColl :MutableCubbieCollection[NC]
  def += (n:N){ insert(n) }
  def ++=(ns:Iterable[N]){ insert(ns) }
  def insert(c:NC) { nodeCubbieColl += c }
  def insert(n:N) {  nodeCubbieColl += cubbify(n) }
  def insert(ns:Iterable[N]) { nodeCubbieColl ++= ns.map(cubbify) }
  def drop:Unit
  def store(nodesToStore:Iterable[N]) {
    val (created, others) = nodesToStore.partition(n => !n.wasDeleted && !n.wasLoadedFromDb)
    nodeCubbieColl ++= created.map(cubbify)
    for(node <- others){
      if(node.wasDeleted){
        nodeCubbieColl.remove(_.idIs(node.id))
      }
      else {
        nodeCubbieColl.updateDelta(_id2cubbie(node.id),cubbify(node))
      }
    }
  }

  def assembleNodes(toAssemble:Seq[N], node2ParentId: Map[N, String], id2Node:Map[String, N]) {

    def assembleHelper(n:N) {
      if(!n.parent.isDefined && node2ParentId.isDefinedAt(n)) {
        val parent = id2Node(node2ParentId(n))
        n.alterParent(Some(parent))(null)
        assembleHelper(parent)
      }
    }

    val mentions = toAssemble.filter(_.isMention)
    for(m <- mentions){
      assembleHelper(m)
    }
  }

}

abstract class MongoNodeCollection[Vars <: NodeVariables[Vars], N <: Node[Vars] with Persistence,
NC<:NodeCubbie[Vars, N]](val names:Seq[String], mongoDB:DB)(implicit ct: ClassTag[Vars]) extends DBNodeCollection[Vars, N, NC]{
  val numBags = ct.runtimeClass.getDeclaredFields.count(_.getType.getName.endsWith("BagOfWordsVariable"))
  assert(names.size == numBags+1, "Insufficient collection names : "+numBags+1+"<"+names.size)
  protected val colls = names.map(mongoDB.getCollection)
  val nodeCubbieColl = new MongoCubbieCollection[NC](colls(0),() => newNodeCubbie,(a:NC) => Seq(Seq(a.parentRef))) with LazyCubbieConverter[NC]
  val varsCubbieColls = colls.tail.map(coll => new MongoCubbieCollection(coll,() => newBOWCubbie,(a:BOWCubbie) => Seq(Seq(a.nodeId))) with LazyCubbieConverter[BOWCubbie])

  def drop: Unit = ???

  def nextBatch(n: Int): Seq[N] = ???

  override def += (n:N){
    var index = 1
    for(v <- n.variables.getVariables){
      v match {
        case bow:BagOfWordsVariable =>
          varsCubbieColls(index) ++= cubbifyBOW(n.id, bow)
          index+=1
        case _ => println("can't cubbify this type")
      }
    }
    insert(n)
  }

  def cubbifyBOW(nodeId:String, bow:BagOfWordsVariable) = bow.value.asHashMap.map{
    case (w,d) => newBOWCubbie.store(nodeId, w, d)
  }

  val WikiTitleExtractor1 = """.+?/wiki/(.+)""".r
  val WikiTitleExtractor2 = """.+?/\.\.\./(.+)""".r
  val WikiTitleExtractor3 = """.+?/(.+)""".r

  def getTitleFromWikiURL(wikiUrl: String): String = {
    val name = wikiUrl match {
      case WikiTitleExtractor1(name) => name
      case WikiTitleExtractor2(name) => name
      case WikiTitleExtractor3(name) => name
      case "" => ""
      case _ => throw new Error("cannot extract wikititle from " + wikiUrl)
    }
    name.replaceAll("_", " ")
  }

  def loadAll: Seq[N] = {
    val node2ParentId = mutable.HashMap[N, String]()
    val id2Node       = mutable.HashMap[String, N]()
    val nodes =
      for(nc <- nodeCubbieColl.toSeq) yield {
        _id2cubbie += nc.id.toString -> nc
        val vars = varsCubbieColls.map(coll => {
          val it = coll.findByAttribute("bid",Seq(nc.id))
          val bag = new BagOfWordsVariable
          for(b <- it){
            bag += (b.word.value, b.count.value)
          }
          bag
        })
        val v = newNodeVars(getTitleFromWikiURL(nc.wikiUrl.value), vars:_*)
        val n = newNode(v,nc)
        id2Node += nc.id.toString -> n
        if(nc.parentRef.isDefined){
          node2ParentId += n -> nc.parentRef.value.toString
        }
        n
      }
    assembleNodes(nodes, node2ParentId.toMap, id2Node.toMap)
    nodes
  }

  //  def loadByIds(ids: Seq[String]): Seq[N] = {
  //    for(nc <- nodeCubbieColl.findByIds(ids)) yield {
  //      val vars = mutable.ArrayBuffer[Var]()
  //      for(coll <- varsCubbieColls){
  //        val it = coll.findByAttribute("nid",Seq(nc.id))
  //        //todo create a var with the cubbies
  //      }
  //      val v = newNodeVars(vars:_*)
  //      val n = newNode(v,n)
  //      n
  //    }
  //  }

  protected def newBOWCubbie : BOWCubbie

  protected def newNodeVars[V <: Var](truth: String, vars: V*) : Vars
}

