
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Airbnb의 csv 데이터를 로드하는 함수들을 구현한 object
  *
  * 각 로드함수의 반환 결과는 DataFrame이다.
  * 
  * (data url : http://insideairbnb.com/get-the-data.html)
  *
  * 2019.06.30 by dhkim
  * */

object Loader {

  // csv 파일 경로 변수들
  val PATH_LISTINGS_DETAIL = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/listings_detail.csv"
  val PATH_LISTINGS = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/listings.csv"
  val PATH_NEIGHBOURHOOD = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/neighbourhoods.csv"
  val PATH_REVIEWS_DETAIL = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/reviews_detail.csv"
  val PATH_REVIEWS = "/Users/dhkdn9192/jupyter/yanolja/data/airbnb/reviews.csv"

  // 1. reviews_detail.csv를 로드하여 DataFrame으로 반환하는 함수
  def loadReviewsDetail(spark: SparkSession): DataFrame = {

    // csv 파일 스키마
    val reviewsDetailSchema = StructType(Seq(
      StructField("listing_id", IntegerType, false),
      StructField("id", IntegerType, false),
      StructField("date", StringType, false),
      StructField("reviewer_id", IntegerType, false),
      StructField("reviewer_name", StringType, false),
      StructField("comments", DoubleType, false)
    ))

    val reviewsDetailDf = spark
      .read
      .schema(reviewsDetailSchema)
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("mode", "DROPMALFORMED")
      .load(PATH_REVIEWS_DETAIL)
      .select(col("listing_id"), col("date"), col("reviewer_id"), col("reviewer_name"))
      .as("reviewsDetailDf")

    /**
      * reviewsDetailDf.show(5)
      * +----------+----------+-----------+-------------+
      * |listing_id|      date|reviewer_id|reviewer_name|
      * +----------+----------+-----------+-------------+
      * |     35303|2011-12-28|    1502908|        Firuz|
      * |     35303|2012-10-01|     350719|       Jordan|
      * |     35303|2013-02-18|    4917704|      Aymeric|
      * |     35303|2013-03-30|    3243253|     Blandine|
      * |     35303|2013-05-01|    1536097|     Kayleigh|
      * +----------+----------+-----------+-------------+
      */
    reviewsDetailDf
  }

  // 2. listings.csv를 로드하여 DataFrame으로 반환하는 함수
  def loadListings(spark: SparkSession): DataFrame = {

    // csv 파일 스키마
    val listingsSchema = StructType(Seq(
      StructField("id", IntegerType, false),
      StructField("name", StringType, false),
      StructField("host_id", IntegerType, false),
      StructField("host_name", StringType, false),
      StructField("neighbourhood_group", StringType, true),
      StructField("neighbourhood", StringType, false),
      StructField("latitude", FloatType, true),
      StructField("longitude", FloatType, true),
      StructField("room_type", StringType, true),
      StructField("price", IntegerType, false),
      StructField("minimum_nights", IntegerType, true),
      StructField("number_of_reviews", IntegerType, true),
      StructField("last_review", StringType, false),
      StructField("reviews_per_month", FloatType, false),
      StructField("calculated_host_listings_count", IntegerType, true),
      StructField("availability_365", IntegerType, true)
    ))

    val listingsDf = spark
      .read
      .schema(listingsSchema)
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("mode", "DROPMALFORMED")
      .load(PATH_LISTINGS)
      .select(col("id"), col("host_id"), col("host_name"), col("neighbourhood"))
      .as("listingsDf")

    /**
      * listingsDf.show(5)
      * +------+-------+-------------------+-------------+
      * |    id|host_id|          host_name|neighbourhood|
      * +------+-------+-------------------+-------------+
      * | 35303| 151977|             Miyuki|   Shibuya Ku|
      * |197677| 964081|    Yoshimi & Marek|    Sumida Ku|
      * |289597| 341577|           Hide&Kei|    Nerima Ku|
      * |370759|1573631|Gilles,Mayumi,Taiki|  Setagaya Ku|
      * |700253| 341577|           Hide&Kei|    Nerima Ku|
      * +------+-------+-------------------+-------------+
      */
    listingsDf
  }

  // 3. neighbourhoods.csv를 로드하여 DataFrame으로 반환하는 함수
  def loadNeighbourhoods(spark: SparkSession): DataFrame = {

    // csv 파일 스키마
    val neighbourhoodsSchema = StructType(Seq(
      StructField("neighbourhood_group", StringType, true),
      StructField("neighbourhood", StringType, false)
    ))

    val neigbourhoodsDf = spark
      .read
      .schema(neighbourhoodsSchema)
      .format("csv")
      .option("header", "true") //first line in file has headers
      .option("mode", "DROPMALFORMED")
      .load(PATH_NEIGHBOURHOOD)
      .drop("neighbourhood_group")
      .withColumn("neighbourhood_id", monotonically_increasing_id())  // 각 지역에 대한 고유 인덱스값 칼럼 추가
      .as("neigbourhoodsDf")

    /**
      * neigbourhoodsDf.show(5)
      * +--------------+----------------+
      * | neighbourhood|neighbourhood_id|
      * +--------------+----------------+
      * |     Adachi Ku|               0|
      * |   Akiruno Shi|               1|
      * |  Akishima Shi|               2|
      * |Aogashima Mura|               3|
      * |    Arakawa Ku|               4|
      * +--------------+----------------+
      */
    neigbourhoodsDf
  }

  // 4. neighbourhood_id(Long) -> neighbourhood_name(String) dictionary
  def getNeighbourhoodMap(spark: SparkSession, neigbourhoodDf: DataFrame) = {
    import spark.implicits._
    val neighbourhoodMap = neigbourhoodDf
      .select(col("neighbourhood_id"), col("neighbourhood"))
      .as[(Long, String)]
      .collect
      .toMap

    neighbourhoodMap
  }

  // 5. reviewer_id(Long) -> reviewer_name(String) dictionary
  def getReviewerMap(spark: SparkSession, reviewsDetailDf: DataFrame) = {
    import spark.implicits._
    val reviewerMap = reviewsDetailDf
      .select(col("reviewer_id"), col("reviewer_name"))
      .as[(Long, String)]
      .collect
      .toMap

    reviewerMap
  }





}
