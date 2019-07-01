
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

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
    recommendations.write.mode(saveMode = SaveMode.Overwrite).parquet(PATH_RESULT_PARQUET)

    // 결과 출력
    recommendations.show(10, false)
    /**
      * +----------+------------+----------------------------------------------------------------------------+----------+
      * |reviewerId|reviewerName|neighbourhoodNames                                                          |date      |
      * +----------+------------+----------------------------------------------------------------------------+----------+
      * |100583296 |靖修          |[Fussa Shi, Ogasawara Mura, Higashikurume Shi, Kokubunji Shi, Hinohara Mura]|2019-07-01|
      * |13959744  |Julien      |[Ogasawara Mura, Tachikawa Shi, Fussa Shi, Shinjuku Ku, Niijima Mura]       |2019-07-01|
      * |37084608  |Barbara     |[Tachikawa Shi, Higashikurume Shi, Inagi Shi, Adachi Ku, Kodaira Shi]       |2019-07-01|
      * |149361088 |Raghav      |[Fussa Shi, Higashikurume Shi, Oshima Machi, Toshima Ku, Kokubunji Shi]     |2019-07-01|
      * |25902656  |Nancy       |[Ogasawara Mura, Hamura Shi, Hinohara Mura, Suginami Ku, Tachikawa Shi]     |2019-07-01|
      * |175806848 |Mohd        |[Fussa Shi, Higashikurume Shi, Oshima Machi, Toshima Ku, Kokubunji Shi]     |2019-07-01|
      * |100134848 |Young       |[Fussa Shi, Ogasawara Mura, Tachikawa Shi, Shinjuku Ku, Kokubunji Shi]      |2019-07-01|
      * |21623616  |Emma        |[Fussa Shi, Ogasawara Mura, Kokubunji Shi, Sumida Ku, Shinjuku Ku]          |2019-07-01|
      * |27406016  |Cheerqiao   |[Ogasawara Mura, Hamura Shi, Hinohara Mura, Suginami Ku, Tachikawa Shi]     |2019-07-01|
      * |196298304 |해호          |[Ogasawara Mura, Niijima Mura, Kokubunji Shi, Minato Ku, Taito Ku]          |2019-07-01|
      * +----------+------------+----------------------------------------------------------------------------+----------+
      */

  }
}
