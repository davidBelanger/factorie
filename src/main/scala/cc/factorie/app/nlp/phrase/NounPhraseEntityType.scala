package cc.factorie.app.nlp.phrase

///** Categorical variable indicating whether the noun phrase is person, location, organization, etc. */
//class NounPhraseEntityType(val phrase:NounPhrase, targetValue:String) extends LabeledCategoricalVariable(targetValue) {
//  def domain = OntonotesNounPhraseEntityTypeDomain
//}
//object OntonotesNounPhraseEntityTypeDomain extends CategoricalDomain[String]{
//  this ++= ner.OntonotesNerDomain.categories
//  this += "MISC"
//}

import cc.factorie._
import cc.factorie.app.nlp._
import cc.factorie.app.nlp.pos._
import cc.factorie.app.nlp.ner.{ConllNerDomain, OntonotesEntityTypeDomain}
import cc.factorie.util.BinarySerializer
import java.io._
import cc.factorie.variable.{LabeledCategoricalVariable, BinaryFeatureVectorVariable, CategoricalVectorDomain, CategoricalDomain}
import cc.factorie.optimize.{PredictorExample, Trainer, OptimizableObjectives}
import cc.factorie.app.classify.backend.LinearMulticlassClassifier
//import cc.factorie.app.nlp.coref.mention.Entity
import cc.factorie.app.nlp.load.LoadConll2011


/** Categorical variable indicating whether the noun phrase is person, location, organization, etc. 
    according to the CoNLL 2003 entity type domain: PER, ORG, LOC, MISC. */
class ConllEntityType(targetValue:String) extends LabeledCategoricalVariable(targetValue) {
  def domain = ConllNerDomain
}
class ConllPhraseEntityType(val phrase:Phrase, targetValue:String) extends ConllEntityType(targetValue)


/** Categorical variable indicating whether the noun phrase is person, location, organization, etc. 
    according to the Ontonotes entity type domain. */
class OntonotesEntityType(targetValue:String) extends LabeledCategoricalVariable(targetValue) {
  def domain = OntonotesEntityTypeDomain
}
class OntonotesPhraseEntityType(val phrase:Phrase, targetValue:String) extends OntonotesEntityType(targetValue)





class OntonotesPhraseEntityTypeLabeler extends DocumentAnnotator {
  def this(stream:InputStream) = { this(); deserialize(stream) }
  def this(file: File) = this(new FileInputStream(file))
  def this(url:java.net.URL) = this(url.openConnection.getInputStream)
  
  object FeatureDomain extends CategoricalVectorDomain[String]
  class FeatureVariable extends BinaryFeatureVectorVariable[String] {
    def domain = FeatureDomain
    override def skipNonCategories: Boolean = domain.dimensionDomain.frozen
  }
  lazy val model = new LinearMulticlassClassifier(OntonotesEntityTypeDomain.size, FeatureDomain.dimensionDomain.size)
  
  def features(mention:Phrase): FeatureVariable = {
    val features = new FeatureVariable
    var tokens = mention.tokens.toSeq
    if (tokens.head.string == "the") tokens = tokens.drop(1)
    if (tokens.length > 0 && tokens.last.string == "'s") tokens = tokens.dropRight(1)
    if (tokens.length == 0) return features // TODO Complain further here? 
    val words = tokens.map(token => cc.factorie.app.strings.collapseDigits(token.string))
    features ++= words
    features += "HEAD="+mention.headToken.string
    features += "LAST="+words.last
    features += "FIRST="+words.last
    mention.tokens.head.prevWindow(3).foreach(token => features += "PREV="+token.string)
    mention.tokens.last.nextWindow(3).foreach(token => features += "NEXT="+token.string)
    for (lexicon <- lexicons) {
      if (lexicon.contains(tokens)) features += "LEX="+lexicon.name
      if (lexicon.containsWord(mention.headToken.string)) features += "HEADLEX="+lexicon.name
    }
    // TODO Add more features
    features
  }
  val lexicons = Seq(
      lexicon.iesl.PersonFirst,
      lexicon.iesl.PersonLast,
      lexicon.iesl.Month,
      lexicon.iesl.PersonHonorific,
      lexicon.iesl.Company,
      lexicon.iesl.Country,
      lexicon.iesl.City,
      lexicon.iesl.AllPlaces,
      lexicon.iesl.USState,
      lexicon.wikipedia.Person,
      lexicon.wikipedia.Event,
      lexicon.wikipedia.Location,
      lexicon.wikipedia.Organization,
      lexicon.wikipedia.ManMadeThing,
      lexicon.wikipedia.Event)
  
  val PersonLexicon = new lexicon.UnionLexicon("NounPhraseEntityTypePerson", lexicon.PersonPronoun, lexicon.PosessiveDeterminer)
  def isWordNetPerson(token:Token): Boolean = wordnet.WordNet.isHypernymOf("person", wordnet.WordNet.lemma(token.string, "NN"))
  def entityTypeIndex(mention:Phrase): Int = {
    if (PersonLexicon.contains(mention) || isWordNetPerson(mention.headToken)) OntonotesEntityTypeDomain.index("PERSON")
    else model.classification(features(mention).value).bestLabelIndex
  }
  def process(mention: Phrase): Unit = mention.attr.getOrElseUpdate(new OntonotesPhraseEntityType(mention, "O")) := entityTypeIndex(mention)
  def process(document:Document): Document = {
    for (mention <- document.attr[NounPhraseList]) process(mention)
    document
  }

  override def tokenAnnotationString(token:Token): String = { val mentions = token.document.attr[NounPhraseList].filter(_.contains(token)); mentions.map(_.attr[OntonotesPhraseEntityType].categoryValue).mkString(",") }
  override def phraseAnnotationString(mention:Phrase): String = { val t = mention.attr[OntonotesPhraseEntityType]; if (t ne null) t.categoryValue else "_" }
  def prereqAttrs: Iterable[Class[_]] = List(classOf[NounPhraseList])
  def postAttrs: Iterable[Class[_]] = List(classOf[OntonotesPhraseEntityType])
 
  def filterTrainingNounPhrases(phrases:Seq[Phrase]): Iterable[Phrase] =
    // TODO This used to filter out phrases corresponding to entities with only one mention, but now we need the Mention to do this. 
    // How important is this filter? -akm
    // mentions.groupBy(m => m.entity).filter(x => x._2.length > 1).map(x => x._2).flatten.filter(mention => !PersonLexicon.contains(mention)) 
    phrases.filter(phrase => !PersonLexicon.contains(phrase))

  def train(trainDocs:Iterable[Document], testDocs:Iterable[Document]): Unit = {
    implicit val random = new scala.util.Random(0)
    val trainNounPhrases = trainDocs.flatMap(_.attr[NounPhraseList])
    FeatureDomain.dimensionDomain.gatherCounts = true
    trainNounPhrases.foreach(features(_))
    FeatureDomain.dimensionDomain.trimBelowCount(3)
    val examples = for (doc <- trainDocs; mention <- filterTrainingNounPhrases(doc.attr[NounPhraseList])) yield
      new PredictorExample(model, features(mention).value, mention.attr[OntonotesPhraseEntityType].intValue, OptimizableObjectives.hingeMulticlass)
    val testNounPhrases = testDocs.flatMap(doc => filterTrainingNounPhrases(doc.attr[NounPhraseList]))
    println("Training ")
    def evaluate(): Unit = {
      println("TRAIN\n"+(new cc.factorie.app.classify.Trial[OntonotesPhraseEntityType,la.Tensor1](model, OntonotesEntityTypeDomain, (t:OntonotesPhraseEntityType) => features(t.phrase).value) ++= trainNounPhrases.map(_.attr[OntonotesPhraseEntityType])).toString)
      println("\nTEST\n"+(new cc.factorie.app.classify.Trial[OntonotesPhraseEntityType,la.Tensor1](model, OntonotesEntityTypeDomain, (t:OntonotesPhraseEntityType) => features(t.phrase).value) ++= testNounPhrases.map(_.attr[OntonotesPhraseEntityType])).toString)
    }
    Trainer.onlineTrain(model.parameters, examples.toSeq, maxIterations=3, evaluate = evaluate)
  }

  // Serialization
  def serialize(filename: String): Unit = {
    val file = new File(filename); if (file.getParentFile ne null) file.getParentFile.mkdirs()
    serialize(new java.io.FileOutputStream(file))
  }
  def deserialize(file: File): Unit = {
    require(file.exists(), "Trying to load non-existent file: '" +file)
    deserialize(new java.io.FileInputStream(file))
  }
  def serialize(stream: java.io.OutputStream): Unit = {
    import cc.factorie.util.CubbieConversions._
    val sparseEvidenceWeights = new la.DenseLayeredTensor2(model.weights.value.dim1, model.weights.value.dim2, new la.SparseIndexedTensor1(_))
    model.weights.value.foreachElement((i, v) => if (v != 0.0) sparseEvidenceWeights += (i, v))
    model.weights.set(sparseEvidenceWeights)
    val dstream = new java.io.DataOutputStream(new BufferedOutputStream(stream))
    BinarySerializer.serialize(FeatureDomain.dimensionDomain, dstream)
    BinarySerializer.serialize(model, dstream)
    dstream.close()  // TODO Are we really supposed to close here, or is that the responsibility of the caller
  }
  def deserialize(stream: java.io.InputStream): Unit = {
    import cc.factorie.util.CubbieConversions._
    val dstream = new java.io.DataInputStream(new BufferedInputStream(stream))
    BinarySerializer.deserialize(FeatureDomain.dimensionDomain, dstream)
    model.weights.set(new la.DenseLayeredTensor2(PennPosDomain.size, FeatureDomain.dimensionDomain.size, new la.SparseIndexedTensor1(_)))
    BinarySerializer.deserialize(model, dstream)
    dstream.close()  // TODO Are we really supposed to close here, or is that the responsibility of the caller
  }

}

object NounPhraseEntityTypeLabeler extends OntonotesPhraseEntityTypeLabeler(cc.factorie.util.ClasspathURL[OntonotesPhraseEntityTypeLabeler](".factorie"))

object NounPhraseEntityTypeLabelerTrainer {
  def main(args:Array[String]): Unit = {
    if (args.length == 0) println("usage: trainfile [modelfile]")
    var trainDocs = LoadConll2011.loadWithParse(args(0), loadSingletons=false, disperseEntityTypes=true)
    val testDocs = trainDocs.takeRight(20)
    trainDocs = trainDocs.dropRight(20)
    val labeler = new OntonotesPhraseEntityTypeLabeler
    for (mention <- labeler.filterTrainingNounPhrases(testDocs.flatMap(_.attr[NounPhraseList])))
      println("%20s  %s".format(mention.attr[OntonotesPhraseEntityType].target.categoryValue, mention.phrase))

    labeler.train(trainDocs, testDocs)
    (trainDocs ++ testDocs).foreach(labeler.process(_))
    for (mention <- labeler.filterTrainingNounPhrases(testDocs.flatMap(_.attr[NounPhraseList])))
      println("%20s %-20s %-20s  %s".format(mention.attr[OntonotesPhraseEntityType].target.categoryValue, mention.attr[OntonotesPhraseEntityType].categoryValue, labeler.isWordNetPerson(mention.headToken).toString, mention.phrase))

    if (args.length > 1) labeler.serialize(args(1))
    
  }
}

