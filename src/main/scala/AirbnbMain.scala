
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * main object
  *
  * Airbnb의 일본 지역 오픈데이터를 활용하여 고객들에게 신규 여행지를 추천하는 추천시스템을 구현한다.
  *
  * 2019.06.30 by dhkim
  */

object AirbnbMain {

  // 파일 경로 변수
  val PATH_ALS_MODEL = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/model"
  val PATH_RESULT_PARQUET = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/resultDf.parquet"

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

    // >> Mean Squared Error = 0.003322032007912435
    println(">> Mean Squared Error = " + mse)

    // load model and make recommendations
    val loadedModel = Recommender.loadModel(sc, PATH_ALS_MODEL)
    val recommendations = Recommender.getRecommendations(spark, loadedModel, 5, reviewerMap, neighbourhoodMap)

    // 최종 결과 DataFrame을 parquet 형태로 저장 (실제 서비스에선 mysql등 db에 저장해야 함)
    recommendations.write.parquet(PATH_RESULT_PARQUET)

    // 결과 출력
    recommendations.show(10, false)
    /**
      * +----------+------------+----------------------------------------------------------------------------+
      * |reviewerId|reviewerName|neighbourhoodNames                                                          |
      * +----------+------------+----------------------------------------------------------------------------+
      * |100583296 |靖修          |[Hamura Shi, Niijima Mura, Higashikurume Shi, Hinohara Mura, Kozushima Mura]|
      * |13959744  |Julien      |[Hamura Shi, Inagi Shi, Niijima Mura, Shinjuku Ku, Kozushima Mura]          |
      * |37084608  |Barbara     |[Inagi Shi, Hinohara Mura, Akishima Shi, Niijima Mura, Higashiyamato Shi]   |
      * |149361088 |Raghav      |[Hamura Shi, Akishima Shi, Toshima Ku, Koganei Shi, Nakano Ku]              |
      * |25902656  |Nancy       |[Hamura Shi, Inagi Shi, Niijima Mura, Tachikawa Shi, Kunitachi Shi]         |
      * |175806848 |Mohd        |[Hamura Shi, Akishima Shi, Toshima Ku, Koganei Shi, Nakano Ku]              |
      * |100134848 |Young       |[Hamura Shi, Niijima Mura, Inagi Shi, Kozushima Mura, Shinjuku Ku]          |
      * |21623616  |Emma        |[Hamura Shi, Niijima Mura, Inagi Shi, Kozushima Mura, Sumida Ku]            |
      * |27406016  |Cheerqiao   |[Hamura Shi, Inagi Shi, Niijima Mura, Tachikawa Shi, Kunitachi Shi]         |
      * |196298304 |해호          |[Akishima Shi, Kunitachi Shi, Inagi Shi, Minato Ku, Higashiyamato Shi]      |
      * +----------+------------+----------------------------------------------------------------------------+
      */

  }
}
