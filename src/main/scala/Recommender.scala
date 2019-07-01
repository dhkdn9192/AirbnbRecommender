
import java.io.File
import java.nio.file.{Files, Paths}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

/**
  * ALS 알고리즘을 사용하여 추천 숙박지역을 계산하는 object
  *
  * load된 dataframe들을 join하여 rating dataframe을 생성한 뒤 ALS 모델을 학습시킨다
  *
  * 2019.06.30 by dhkim
  */

object Recommender {

  // 1. ALS 모델을 학습시키기 위한 RDD[Rating] 데이터 생성 함수
  def getRating(spark: SparkSession, listingsDf: DataFrame, neigbourhoodsDf: DataFrame, reviewsDetailDf: DataFrame) = {

    import spark.implicits._

    // listings 숙소 목록에 지역 id칼럼이 추가되도록 join
    val joinedListingNeighbourDf = listingsDf
      .join(neigbourhoodsDf, col("listingsDf.neighbourhood") === col("neigbourhoodsDf.neighbourhood"), "inner")
      .drop(col("neigbourhoodsDf.neighbourhood"))  // 같은 칼럼이 이미 있으므로 한 쪽은 drop
      .as("joinedListingNeighbourDf")

    /**
      * println(s">> joinedListingNeighbourDf count: ${joinedListingNeighbourDf.count()}")
      * >> joinedListingNeighbourDf count: 10014
      *
      * joinedListingNeighbourDf.show(5)
      * +------+-------+-------------------+-------------+----------------+
      * |    id|host_id|          host_name|neighbourhood|neighbourhood_id|
      * +------+-------+-------------------+-------------+----------------+
      * | 35303| 151977|             Miyuki|   Shibuya Ku|              52|
      * |197677| 964081|    Yoshimi & Marek|    Sumida Ku|              56|
      * |289597| 341577|           Hide&Kei|    Nerima Ku|              43|
      * |370759|1573631|Gilles,Mayumi,Taiki|  Setagaya Ku|              51|
      * |700253| 341577|           Hide&Kei|    Nerima Ku|              43|
      * +------+-------+-------------------+-------------+----------------+
      */


    // listings df에 reviewer 정보가 포함되도록 join
    val joinedListingReviewsDf = joinedListingNeighbourDf
      .join(reviewsDetailDf, col("joinedListingNeighbourDf.id") === col("reviewsDetailDf.listing_id"), "inner")
      .drop("id")
      .as("joinedListingReviewsDf")

    /**
      * println(s">> joinedListingReviewsDf count: ${joinedListingReviewsDf.count()}")
      * >> joinedListingReviewsDf count: 256602
      *
      * joinedListingReviewsDf.show(5)
      * +--------+---------+-------------+----------------+----------+----------+-----------+-------------+
      * | host_id|host_name|neighbourhood|neighbourhood_id|listing_id|      date|reviewer_id|reviewer_name|
      * +--------+---------+-------------+----------------+----------+----------+-----------+-------------+
      * |57359337|Youngjoon|    Meguro Ku|              34|  11851618|2016-07-11|    8224354|      Vanessa|
      * |57359337|Youngjoon|    Meguro Ku|              34|  11851618|2016-07-13|   79181639|           우정|
      * |57359337|Youngjoon|    Meguro Ku|              34|  11851618|2016-07-30|   81013810|           선희|
      * |57359337|Youngjoon|    Meguro Ku|              34|  11851618|2016-08-01|   81407184|   P Kyungeun|
      * |57359337|Youngjoon|    Meguro Ku|              34|  11851618|2016-08-04|   43043351|      Marsela|
      * +--------+---------+-------------+----------------+----------+----------+-----------+-------------+
      */


    val rating = joinedListingReviewsDf
      .groupBy("reviewer_id", "reviewer_name", "neighbourhood_id", "neighbourhood")
      .count()
      .rdd
      .map(r => Rating(
        r.getAs[Int]("reviewer_id"),
        r.getAs[Long]("neighbourhood_id").toInt,
        r.getLong(4).toDouble
      ))

    /** rating
      * +-----------+-------------+--------------+
      * |         _1|           _2|            _3|
      * +-----------+-------------+--------------+
      * |    5752694|           52|             2|
      * |   42793753|           58|             2|
      * |  101608784|           54|             2|
      * |   41768546|           42|             2|
      * |  168061498|           51|             2|
      * +-----------+-------------+--------------+
      */

    rating
  }


  // 2. ALS 모델을 학습시키는 함수
  def trainModel(sc: SparkContext, rating: RDD[Rating], numIterations: Int, path: String) = {
    // val Array(training, test) = rating.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS
    val rank = 10
    val model = ALS.train(rating, rank, numIterations, 0.01)

    // Evaluate the model on rating data
    val usersProducts = rating
      .map { case Rating(user, product, rate) => (user, product) }
    val predictions = model
      .predict(usersProducts)
      .map { case Rating(user, product, rate) => ((user, product), rate) }
    val ratesAndPreds = rating
      .map { case Rating(user, product, rate) => ((user, product), rate) }
      .join(predictions)

    // MSE 계산
    val MSE = ratesAndPreds
      .map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
      }
      .mean()

    // 학습된 모델 저장. 이미 해당 경로에 모델이 저장되어 있으면 삭제
    if (Files.exists(Paths.get(path))) {
      FileUtils.deleteQuietly(new File(path))
    }
    model.save(sc, path)

    // 모델 MSE값 반환
    MSE

  }


  // 3. 저장된 모델 로드 함수
  def loadModel(sc: SparkContext, path: String) =
    MatrixFactorizationModel.load(sc, path)


  // 4. 학습된 ALS 모델을 사용하여 고객별 여행추천지를 계산하는 함수
  def getRecommendations(spark: SparkSession, model: MatrixFactorizationModel, products: Int,
                         reviewerMap: Map[Long, String], neighbourhoodMap: Map[Long, String]) = {
    val recommendationsRdd = model
      .recommendProductsForUsers(products)
      .map(r => {
        val reviewerId = r._1.toInt
        val reviewerName = reviewerMap.getOrElse(reviewerId.toLong, "empty")
        val neighbourhoodNames = r._2.map(rating => neighbourhoodMap.getOrElse(rating.product.toLong, "empty")).toList
        Row(reviewerId, reviewerName, neighbourhoodNames)
      })  // RDD[Row]

    // RDD를 DataFrame으로 변환하기 위한 스키마 정의
    val schema = new StructType()
      .add(StructField("reviewerId", IntegerType, true))
      .add(StructField("reviewerName", StringType, true))
      .add(StructField("neighbourhoodNames", ArrayType(StringType), true))

    // 오늘 날짜를 생성하는 udf
    val nowDatetimeUdf = udf(() => DateTimeFormatter.ofPattern("yyyy-MM-dd").format(LocalDateTime.now))

    // RDD[Row]를 DataFrame으로 변환
    val recommendationsDf = spark
      .createDataFrame(recommendationsRdd, schema)
      .withColumn("date", nowDatetimeUdf())

    /**
      * recommendationsDf.show()
      *
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

    recommendationsDf
  }


}
