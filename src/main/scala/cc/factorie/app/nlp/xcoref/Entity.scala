//package cc.factorie.app.nlp.xcoref
//
//import cc.factorie.app.nlp.coref.mention.Mention
//import cc.factorie.app.nlp.Document
//import cc.factorie.util.coref.GenericEntityMap
//
//
//class WithinDocEntities(val entityMap: GenericEntityMap){
//  //this makes a bunch of WithinDocEntity objects based on the entity map. It also resolves their type, etc.
//  def entities: Iterable[WithinDocEntity]
//
//}
//
//
//class WithinDocEntity(val mentions: Iterable[Mention]) {
//  def mentions: Iterable[Mention]
//  def isNamed: Boolean
//  def canonicalName: String
//  def id: String
//  def gender: String
//  def canonicalMention : String
//  def crossDocId: String
//  def crossDocEntity: CrossDocEntity
//  def doc: Document
//  def entityType : String
//
//  mentions.foreach(m => m.attr += this)
//}
//
//trait CrossDocEntity {
//  def withinDocEntityIds: Iterable[String]
//  def isNamed: Boolean
//  def canonicalName: String
//  def id: String
//  def gender: String
//  def entityType: String
//}
//
