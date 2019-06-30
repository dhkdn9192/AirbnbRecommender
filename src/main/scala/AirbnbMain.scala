
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * main object
  *
  * 목표: Airbnb의 일본 지역 오픈데이터를 활용하여 고객들에게 신규 여행지를 추천하는 추천시스템을 구현한다.
  *
  */

object AirbnbMain {

  val PATH_ALS_MODEL = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/model"

  def main(args: Array[String]): Unit = {

    // SparkContext 생성
    val conf: SparkConf = new SparkConf()
    conf.setMaster("local[*]") // local 환경에서 수행
    conf.setAppName("AirbnbRecommender") // app 이름
    conf.set("spark.driver.bindAddress", "127.0.0.1") // driver ip
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.kryoserializer.buffer.max", "128m")
    val sc = new SparkContext(conf)

    // SparkSession 생성
    val spark = SparkSession
      .builder()
      .appName("AirbnbRecommender")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    // load dataframe
    val reviewsDetailDf = Loader.loadReviewsDetail(spark)
    val listingsDf = Loader.loadListings(spark)
    val neigbourhoodsDf = Loader.loadNeighbourhoods(spark)

    // id2name Map
    val neighbourhoodMap = Loader.getNeighbourhoodMap(spark, neigbourhoodsDf)
    val reviewerMap = Loader.getReviewerMap(spark, reviewsDetailDf)
    sc.broadcast(neighbourhoodMap)  // broadcast to executors
    sc.broadcast(reviewerMap)       // broadcast to executors

    // ALS recommendations
    val rating = Recommender.getRating(spark, listingsDf: DataFrame, neigbourhoodsDf: DataFrame, reviewsDetailDf: DataFrame)
    val mse = Recommender.trainModel(sc, rating, 3, PATH_ALS_MODEL)
    println(">> Mean Squared Error = " + mse)
    val loadedModel = Recommender.loadModel(sc, PATH_ALS_MODEL)
    val recommendations = Recommender.getRecommendations(spark, loadedModel, 5, reviewerMap, neighbourhoodMap)



    recommendations.show(10, false)





  }
}
