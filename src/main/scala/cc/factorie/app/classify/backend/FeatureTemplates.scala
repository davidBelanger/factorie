package cc.factorie.app.classify.backend

import cc.factorie.variable._
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator, DenseTensor2, Tensor1}
import cc.factorie.model.{Weights2, Parameters}


class FeatureTemplateVectorVariable(val template: FeatureTemplate[_], val domain: CategoricalVectorDomain[String])  extends FeatureVectorVariable[String]
class HashedFeatureTemplateVectorVariable(val template: FeatureTemplate[_], val domain: DiscreteDomain)  extends cc.factorie.variable.HashFeatureVectorVariable

//templates have no state. They do not have associated domains. You pass a template to a domain, and it returns a feature vector that points to this domain.
trait FeatureTemplateDomain {
  def domain: VectorDomain
  def newFeatureVector(template: FeatureTemplate[_]): FeatureVectorVar[String]
}

class CategoricalFeatureTemplateDomain extends FeatureTemplateDomain {
  object domain extends CategoricalVectorDomain[String]
  def newFeatureVector(template: FeatureTemplate[_]) = new FeatureTemplateVectorVariable(template,domain)
}

class HashedFeatureTemplateDomain(size: Int) extends FeatureTemplateDomain {
  object domain extends DiscreteDomain(size)
  def newFeatureVector(template: FeatureTemplate[_]) = new HashedFeatureTemplateVectorVariable(template,domain)
}

//A feature template is basically something that has a name and method that takes a variable and returns strings representing binary features
//The template has no state; it is a factory for features. It has methods getAndUpdateFeatures and addFeatureVector for adding features to variables.
trait FeatureTemplate[V]{
  //these next two methods are the only things you need to implement
  def name: String
  def computeFeatures(v: V, ftv: FeatureTemplateVariable[V]): Seq[String]
  //

  //this computes the features and caches their strings (in input_variable.map) for future use
  def getAndUpdateFeatures(v: V, ftv: FeatureTemplateVariable[V]): Seq[String] = {
    ftv.map.getOrElseUpdate(this, computeFeatures(v, ftv))
  }

  //this computes the features and adds them to the input variable as a proper factorie FeatureVectorVar
  def addFeatureVector(v: V, ftv: FeatureTemplateVariable[V], d: FeatureTemplateDomain): Unit = {
    val f = d.newFeatureVector(this)
    f ++= computeFeatures(v, ftv)
    ftv.featureVectorMap(this) = f
  }
}

//This is basically a map from feature templates to the features computed w.r.t to these templates
class FeatureTemplateVariable[V]{
  def templates = map.keys
  def allFeatures = map.values
  val map = collection.mutable.HashMap[FeatureTemplate[V],Seq[String]]()    //these strings are useful to keep explicitly for the sake of caching (when some templates depend on others)
  val featureVectorMap = collection.mutable.HashMap[FeatureTemplate[V],FeatureVectorVar[String]]()  //this is what is used when actually scoring FeatureTemplateVariables with a model

  override def toString() = {
    featureVectorMap.map(kv => "template:" + kv._1.name + " " + kv._2.toString()).mkString(";")
  }
}



//This takes two templates and returns the cross product of all their features. note that it uses cached computation from these templates
class CrossProductTemplate[V](t1: FeatureTemplate[V], t2: FeatureTemplate[V]) extends FeatureTemplate[V] {
  def name: String = t1.name+"&"+t2.name
  def computeFeatures(v: V, ftv: FeatureTemplateVariable[V]): Seq[String] = {
    for (f1 <- t1.getAndUpdateFeatures(v, ftv); f2 <- t2.getAndUpdateFeatures(v, ftv)) yield f1+"&"+f2
  }
}


trait TemplateModel[V] extends OptimizablePredictor[Tensor1,FeatureTemplateVariable[V]] {
  def templates: Seq[(FeatureTemplate[V], FeatureTemplateDomain)]

  //Use this if you have feature templates that use cached computation from previous templates. If you are using this, make
  //sure to order your templates from early to latter.
  //This requires extra memory and some copying, but the savings due to caching are more important
  def getFeatureVectors(v: V): FeatureTemplateVariable[V] = {
    val ftv = new FeatureTemplateVariable[V]
    templates.foreach(t => t._1.getAndUpdateFeatures(v, ftv))
    templates.foreach {case (t,d) =>
      val f = d.newFeatureVector(t)
      f ++= t.getAndUpdateFeatures(v, ftv)
      ftv.map.remove(t)
      ftv.featureVectorMap(t) = f
    }
    ftv.map.clear()
    ftv
  }
  //use this if your feature templates don't depend on each other and you are not using any caching
  def getFeatureVectorsNonCached(v: V): FeatureTemplateVariable[V] = {
    val ftv = new FeatureTemplateVariable[V]
    templates.foreach {case (t,d) =>
      val f = d.newFeatureVector(t)
      f ++= t.computeFeatures(v, ftv)
      ftv.featureVectorMap(t) = f
    }
    ftv
  }
}

//This is just like a standard MulticlassModel, except that your weights are each associated with a feature template You pass in the domains as a constructor argument, so that you can choose
//which ones to have hashed domains and which ones to have explicit domains
class MulticlassTemplateModel[V](inTemplates: Seq[(FeatureTemplate[V], FeatureTemplateDomain)], outputSize: Int) extends TemplateModel[V] with Parameters with MulticlassClassifier[FeatureTemplateVariable[V]]{
  val templatesWithWeights: Seq[(FeatureTemplate[V], FeatureTemplateDomain, Weights2)] = inTemplates.map(t => (t._1,t._2,Weights(new DenseTensor2(t._2.domain.dimensionSize, outputSize))))
  val templates = templatesWithWeights.map { case (a,b,c) => (a,b)}
  def accumulateStats(accumulator: WeightsMapAccumulator, features: FeatureTemplateVariable[V], gradient: Tensor1) {
    templatesWithWeights.foreach { case (t,d,w) => accumulator.accumulate(w, features.featureVectorMap(t).value outer gradient)}
  }
  def score(features: FeatureTemplateVariable[V]): Tensor1 = {
    val v0 = new DenseTensor1(outputSize)
    templatesWithWeights.foreach { case (t,_,w) => v0 += w.value.leftMultiply(features.featureVectorMap(t).value) }
    v0
  }

}

class CategoricalTemplateModel[V](templates: Seq[FeatureTemplate[V]], outputSize: Int) extends MulticlassTemplateModel[V](templates.map(t => (t,new CategoricalFeatureTemplateDomain)), outputSize)

//use this in any model that has a bias
class BiasFeatureTemplate[V] extends FeatureTemplate[V] {
  def name = "bias"
  def computeFeatures(v: V, ftv: FeatureTemplateVariable[V]) = Seq("bias")
}

