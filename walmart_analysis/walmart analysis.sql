SELECT * FROM walmart;
SELECT COUNT(*) FROM walmart;
SELECT DISTINCT payment_method FROM walmart;
-- Because invoice_id is unique, market basket analysis cannot be performed.
-- The data does not have a basket structure (multiple products per transaction)

-- 1. Branch Performance Analysis
-- 1a. Which branch contributes the highest percentage of total company sales revenue?
SELECT SUM(total) AS total_sales
FROM walmart; -- hasilnya single line, jumlah value dari kolom total (keseluruhan )
SELECT 
    branch,
    ROUND(SUM(total), 2) AS branch_total,
    ROUND((100 * SUM(total) / (SELECT SUM(total) FROM walmart)), 2) AS contribution_percent
FROM walmart
GROUP BY branch
ORDER BY branch_total DESC;
-- Answer: WALM09 has the highest contribution percentage, with branch_total = 25,688.34 and contribution_percent = 2.12%

-- 1b. Which branch shows the most consistent monthly performance (lowest variance in total sales)?
SELECT branch,
	VARIANCE(monthly_total) AS sales_variance
FROM(
	SELECT branch,
	EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month,
	SUM(total) AS monthly_total
	FROM walmart
	GROUP BY branch, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
) AS monthly_sales
GROUP BY branch
ORDER BY sales_variance ASC
LIMIT 1;
-- Answer: WALM041 shows the most consistent monthly performance because it has the lowest sales variance (1904240.48) compared to other branches.

-- 1c. Which branches have the highest total sales per transaction?
WITH branch_metrics AS (
SELECT branch, SUM(total) AS branch_total, COUNT(invoice_id) AS transaction
FROM walmart
GROUP BY branch
), branch_transactions AS(
SELECT *, branch_total/transaction AS total_per_transaction
FROM branch_metrics 
GROUP BY branch 
)
SELECT branch, branch_total, transaction, total_per_transaction
FROM branch_transactions 
ORDER BY total_per_transaction DESC;
-- Answer: Branch WALM07 has the highest total sales per transaction, with branch_total = 69,438.55, transaction_count = 380, and total_per_transaction = 182.73

-- 1d. Are branches with higher average ratings also more profitable on average?
WITH branch_metrics AS(
SELECT branch, AVG(rating) AS avg_rating, AVG(profit_margin*total) AS avg_profit
FROM walmart 
GROUP BY branch
)
SELECT *
FROM branch_metrics
ORDER BY avg_rating DESC; 
-- Answer: No, there is no correlation between average ratings and average profit.
-- Branch WALM076 has an average rating of 6.81 but lower average profit (36.40),
-- while branch WALM059 has a slightly lower average rating (6.73) but higher average profit (48.60).
-- This suggests that customers give ratings based on their own personal experience and objectivity, not on store profitability.

-- 1e. “Which branches show a decrease in profit margin despite increased total sales compared to the previous month?”
WITH monthly_sales AS(
SELECT branch, SUM(total) AS total_sales, AVG(profit_margin) AS avg_profit_margin, 
EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month 
FROM walmart
GROUP BY branch, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
), lag_monthly_sales AS(
SELECT *, LAG(total_sales) OVER(PARTITION BY branch ORDER BY month) AS prev_total_sales, LAG(avg_profit_margin) OVER(PARTITION BY branch ORDER BY month) AS prev_avg_profit_margin
FROM monthly_sales
)
SELECT *
FROM lag_monthly_sales 
WHERE avg_profit_margin < prev_avg_profit_margin
AND total_sales > prev_total_sales 
ORDER BY branch, month;
-- Answer: From the output, we can see branches that have a decreasing average profit margin compared to the previous month, even though total sales may be increasing.
-- Branches WALM001 and WALM097 appear multiple times across different months,
-- which suggests these two branches may have operational issues that contribute to the inconsistency.

-- 1f. Which branches show declining profitability over time even if total revenue increases?
WITH branch_sales AS(
SELECT branch, SUM(profit_margin*total) AS total_profit, SUM(total) AS total_revenue, 
EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month
FROM walmart
GROUP BY branch, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
), lag_branch_sales AS(
SELECT *, LAG(total_profit) OVER(PARTITION BY branch ORDER BY month) AS prev_total_profit, LAG(total_revenue) OVER(PARTITION BY branch ORDER BY month) AS prev_total_revenue 
FROM branch_sales 
)
SELECT *
FROM lag_branch_sales
WHERE total_profit < prev_total_profit 
AND total_revenue > prev_total_revenue
ORDER BY branch, month;
-- Answer: From the output, WALM051 and WALM096 show this issue. Branch WALM051 experiences it in month 5 (May), and branch WALM096 in month 9 (September).
-- This suggests that the products frequently purchased by customers may generate lower profit even though the number of items sold increases.
-- My hypothesis is that the issue may be related to seasonal discounts, such as summer sales for WALM051 or fall discounts for WALM096,
-- which reduce the profit margin despite higher sales volume.

-- 1g. Which branch would benefit most from weekend promotions?
WITH traffic_sales AS(
SELECT branch, 
CASE 
   WHEN DAYOFWEEK(STR_TO_DATE(date, '%d/%m/%y')) IN (6,7) THEN 'weekend'
   ELSE 'weekday'
END day_type,
SUM(total) AS total_sales, AVG(rating) AS avg_rating, COUNT(invoice_id) AS freq
FROM walmart
GROUP BY branch, 
CASE 
   WHEN DAYOFWEEK(STR_TO_DATE(date, '%d/%m/%y')) IN(6,7) THEN 'weekend'
   ELSE 'weekday'
END
), pivoted_sales AS(
SELECT branch, 
MAX(CASE WHEN day_type = 'weekend' THEN total_sales END) AS weekend_sales,
MAX(CASE WHEN day_type = 'weekday' THEN total_sales END) AS weekday_sales
FROM traffic_sales
GROUP BY branch
)
SELECT branch, weekend_sales, weekday_sales, 
(weekend_sales - weekday_sales) AS weekend_diff,
(weekend_sales - weekday_sales) / weekday_sales AS weekend_gain,
CASE 
	WHEN (weekend_sales - weekday_sales) / weekday_sales >= 0.25 THEN 'qualified'
	ELSE 'not qualified'
END AS promotion_status
FROM pivoted_sales 
ORDER BY weekend_gain DESC;
-- Answer: Weekend promotions would only make sense if weekend sales earn more than weekday sales.
-- However, due to the imbalance in the number of days (5 weekdays vs 2 weekend days), there may be no branch that qualifies for promotion status based on total sales.
-- From the output, we can see that no branch has a positive weekend gain that would qualify for a promotion.
-- If we change the calculation from total sales to average sales per day, some branches might show positive weekend gain and become qualified for weekend promotions.

-- 2. Category Performance Analysis
-- 2a. Which category contributes the highest percentage of total company sales revenue?
SELECT category,
ROUND(SUM(total),2) AS category_total,
ROUND((100*SUM(total)/(SELECT SUM(total) FROM walmart)),2) AS contribution_percent 
FROM walmart
GROUP BY category 
ORDER BY category_total DESC;
-- Answer: Fashion accessories contributes 40.46% and Home and Lifestyle contributes 40.44% of total revenue.
-- These two categories have almost the same contribution value.

-- 2b. Which product categories have accelerating or decelerating monthly growth rates over time?
WITH monthly_sales AS(
SELECT category, SUM(total) AS cat_monthly_total, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month
FROM walmart
GROUP BY category, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
), growth_monthly AS (
SELECT category, month, cat_monthly_total,
LAG(cat_monthly_total) OVER(PARTITION BY category ORDER BY month) AS prev_cat_monthly_total
FROM monthly_sales
), growth_rate AS(
SELECT category, month,
(cat_monthly_total - prev_cat_monthly_total)/ prev_cat_monthly_total AS growth_percentage 
FROM growth_monthly 
WHERE prev_cat_monthly_total IS NOT NULL
)
SELECT * 
FROM growth_rate
GROUP BY category, month;
-- Answer: Electronic accessories shows an outlier in month 11 (November) with 38.52% growth from the previous month, but this growth did not last into the next month (only 0.028% growth).
-- Fashion accessories also shows growth from the previous month, but then dropped into negative percentage.

-- 2c. Which categories that generate high sales but have below-average profit margins?
WITH category_sales AS(
SELECT category, SUM(total) AS category_total, 
AVG(profit_margin) AS avg_cat_margin
FROM walmart 
GROUP BY category
), 
overall_profit_margin AS(
SELECT AVG(profit_margin) AS avg_overall_margin
FROM walmart
),
avg_overall_sales AS(
SELECT AVG(total) AS avg_overall_sales
FROM walmart
)
SELECT c.category, 
ROUND(c.category_total, 2) AS category_total, 
ROUND(c.avg_cat_margin, 4) AS avg_cat_margin, 
ROUND(o.avg_overall_margin, 4) AS avg_overall_margin,
CASE WHEN c.category_total > a.avg_overall_sales 
AND c.avg_cat_margin < o.avg_overall_margin
	THEN 'high sales but low profit margin'
        ELSE 'normal'
    END AS status
FROM category_sales c
CROSS JOIN overall_profit_margin o
CROSS JOIN avg_overall_sales a
ORDER BY c.category_total DESC;
-- Answer: Fashion accessories, Electronic accessories, and Sports and travel are the categories with high sales but below-average profit margins.

-- 2d. Which category yields the highest profit per unit sold?
WITH metrics AS(
SELECT category, COUNT(invoice_id) AS unit_sold, 
SUM(profit_margin*total) AS cat_total_profit
FROM walmart 
GROUP BY category)
SELECT category, cat_total_profit/unit_sold AS profit_per_unit
FROM metrics
ORDER BY profit_per_unit DESC;
-- Answer: Sports and travel has the highest profit per unit (124.18), followed by Food and beverages (123.87), and Health and beauty (122.84).


-- 2e. Which combination of city and category produces the highest profit density (profit per transaction)?
WITH metrics AS(
SELECT category, city, COUNT(invoice_id) AS transactions, 
SUM(profit_margin*total) AS total_profit
FROM walmart
GROUP BY city, category)
SELECT city, category, total_profit/transactions AS profit_per_transaction
FROM metrics
ORDER BY profit_per_transaction DESC;
-- Answer: The combination with the highest profit density is McKinney + Sports and travel, with 458.11 profit per transaction.
-- The second highest is Pharr + Sports and travel with 428.72 profit per transaction.
-- The third highest is Waco + Food and beverages with 426.11 profit per transaction.

-- 3. City and Time-Based Analysis 
-- 3a. Which city contributes the highest percentage of total company sales revenue?
WITH city_sales AS(
SELECT city, SUM(total) AS city_total
FROM walmart 
GROUP BY city
), overall AS(
SELECT SUM(total) AS total_revenue 
FROM walmart
)
SELECT c.city, c.city_total,
c.city_total / o.total_revenue * 100 AS percentage_contribution
FROM city_sales c
CROSS JOIN overall o
ORDER BY percentage_contribution DESC;
-- Answer: Weslaco has the highest revenue contribution among all Walmart cities, with a city total of 231,758.95.
-- However, percentage contribution does not have a specific impact because no single city dominates with more than 10% contribution. This suggests that buying power across Walmart cities tends to be evenly distributed.
-- Even though Weslaco's contribution is the highest, it is still not close to dominating the overall revenue share.

-- 3b. How does the average profit margin differ across cities?
SELECT city, AVG(profit_margin) AS avg_profit_margin
FROM walmart 
GROUP BY city
ORDER BY avg_profit_margin DESC;
-- Answer: The average profit margin across cities shows that Mansfield and New Braunfels have profit margins above 0.5 (but not reaching 0.58).
-- This indicates that operations are already good and pricing distribution tends to be competitive across branches in these cities.
-- The lowest average profit margins (0.1799) are found across three cities: Alice, Canyon, and Mineral Wells — all below 0.2, which is less than half of the typical average (0.4).
-- These three cities should be investigated further to identify operational or pricing issues.

-- 3c. Do cities with more transactions always generate higher total revenue per transaction?
WITH metrics AS(
SELECT city, SUM(total) AS city_total, COUNT(invoice_id) AS transaction, SUM(total)/COUNT(invoice_id) AS city_revenue_per_transaction
FROM walmart
GROUP BY city)
SELECT *
FROM metrics
ORDER BY city_revenue_per_transaction DESC;
-- Answer: The city with the highest revenue per transaction is McKinney, with only 380 transactions, while Weslaco has 1,980 transactions and revenue per transaction of 117.04.
-- This suggests that overall buying power in Weslaco is more generalized than in McKinney. The higher average in McKinney could be influenced by a few outlier transactions with higher values.
-- Therefore, when a city shows high revenue per transaction, the total transaction volume should also be investigated to determine if the average is driven by a small number of high-value purchases.

-- 3d. Which months show the strongest sales growth compared to the previous month?
WITH monthly_sales AS(
SELECT SUM(total) AS month_total, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month 
FROM walmart
GROUP BY EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
), growth_month AS(
SELECT month, month_total, LAG(month_total) OVER(ORDER BY month) AS prev_month_total
FROM monthly_sales
), growth_rate AS(
SELECT month, month_total, 
(month_total - prev_month_total)/prev_month_total AS growth_percentage 
FROM growth_month 
WHERE prev_month_total IS NOT NULL)
SELECT *
FROM growth_rate
ORDER BY growth_percentage DESC;
-- Answer: The growth percentage in month 11 (November) is 2.28%, which is more than double the second-highest growth month.
-- The second-highest growth is in month 8 (August) at 1.37% growth from July.
-- This highlights that August and November are peak growth months, suggesting a recurring customer pattern.
-- Sales can be strategically increased during these specific months to take advantage of this trend.

-- 3e. Are there any seasonal patterns (e.g., specific months with repeated peaks or dips)?
WITH month_metrics AS(
SELECT AVG(total) AS avg_month, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month 
FROM walmart 
GROUP BY EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) 
), overall AS(
SELECT AVG(total) AS avg_revenue
FROM walmart)
SELECT m.*, m.avg_month/o.avg_revenue AS seasonal_index
FROM month_metrics m
CROSS JOIN overall o
ORDER BY seasonal_index DESC;
-- Answer: Using the seasonal index as an indicator, most monthly sales are in a dip situation (below average).
-- However, the first three months of the year show peak purchasing activity.
-- Right after Christmas, there are multiple consecutive holiday seasons over three months that drive higher sales.

-- 3f. Which hour of the day generates the highest total revenue per branch?”
-- total revenue per hour per branch
SELECT branch, EXTRACT(HOUR FROM time) AS hour, SUM(total) AS total_per_branch_hour
FROM walmart
GROUP BY branch, hour 
ORDER BY total_per_branch_hour DESC;
-- Answer: The highest total revenue per branch hour is at WALM074 during the 16:00 hour, generating 19,729.1. This is slightly above the median but still a good hour for operations.
-- In contrast, WALM078 during the 21 hour generates only 1870 in total revenue per branch hour.

-- 3g. Does sales efficiency (sales per transaction) differ across employee shifts?
WITH metrics AS(
SELECT COUNT(invoice_id) AS transaction, SUM(total) AS sales, SUM(total)/COUNT(invoice_id) AS sales_per_transaction, EXTRACT(HOUR FROM time) AS hour
FROM walmart
GROUP BY EXTRACT(HOUR FROM time)
), shifted AS(
SELECT *, 
CASE 
    WHEN hour BETWEEN 4 AND 9 THEN 'morning_shift'
    WHEN hour BETWEEN 10 AND 14 THEN 'midday_shift'
    WHEN hour BETWEEN 15 AND 21 THEN 'evening_shift'
    ELSE 'midnight_shift'
END AS employee_shift 
FROM metrics)
SELECT *
FROM shifted
ORDER BY sales_per_transaction DESC;
-- Answer: Sales efficiency does vary across employee shifts and working hours.
-- The midday shift has the highest sales per transaction, while morning and evening shifts do not show the same values across their working hours.
-- The lowest sales per transaction occurs during the midnight shift, which operates for 23 hours but only generates 87.67 per transaction.
-- This suggests that midnight and morning shifts do not require as many customers, and employee working hours show no correlation with sales performance.
-- Other factors — such as customer service quality or store layout — may need to be investigated for improvement.


-- 3h. Are weekend sales consistently higher than weekday sales?”
WITH week_metrics AS(
SELECT total,
CASE
	WHEN DAYOFWEEK(STR_TO_DATE(date, '%d/%m/%y')) IN (6,7) THEN 'weekend'
	ELSE 'weekday'
END AS day_type
FROM walmart)
SELECT day_type, COUNT(*) AS transaction_count, ROUND(AVG(total),2) AS avg_sales 
FROM week_metrics
GROUP BY day_type
ORDER BY day_type;
-- Answer: Weekday transactions (across 5 days) total 35,615 with an average sales value of 120.17. This is only twice the transaction volume of weekend days (which only has 2 open days).
-- The average sales on weekends is actually higher than on weekdays. Essentially, weekend sales need to cover the operational costs for those two days, while the profit from weekend sales becomes gross margin for the company.

-- 3i. During which period of the day do customers spend the most per purchase?”
-- most spending per purchase by time period
WITH hourly_metrics AS(
SELECT SUM(total) AS hourly_total, COUNT(invoice_id) AS transaction, EXTRACT(HOUR FROM time) AS hour
FROM walmart
GROUP BY hour
), metrics AS(
SELECT hourly_total/transaction AS spending_per_purchase,
CASE 
      WHEN hour BETWEEN 4 AND 9 THEN 'morning'
      WHEN hour BETWEEN 10 AND 14 THEN 'midday'
      WHEN hour BETWEEN 15 AND 21 THEN 'evening'
      ELSE 'midnight'
END AS time_period
FROM hourly_metrics
)
SELECT * 
FROM metrics 
ORDER BY spending_per_purchase DESC;
-- Answer: Customer spending per purchase is dominated by the midday and evening shifts, meaning those periods already drive most of the revenue.
-- Management should investigate the maximum benchmarks for midday and evening shifts. If sales cannot be increased further during peak hours, then morning and midnight shifts need to be re-strategized so business profit is not consumed by operational costs during busy hours.
-- At a minimum, sales from morning and midnight shifts should cover their own operational costs. This way, profits generated during midday and evening are not used to subsidize other shifts.

-- 3j. Which operational shift contributes most to overall company revenue?
WITH metrics AS(
SELECT 
CASE
	WHEN EXTRACT(HOUR FROM time) BETWEEN 4 AND 9 THEN 'morning'
	WHEN EXTRACT(HOUR FROM time) BETWEEN 10 AND 14 THEN 'midday'
	WHEN EXTRACT(HOUR FROM time) BETWEEN 15 AND 21 THEN 'evening'
	ELSE 'midnight'
END AS operational_shift,
SUM(total) AS shift_revenue
FROM walmart
GROUP BY operational_shift
), total_revenue AS(
SELECT sum(total) AS grand_total 
FROM walmart
)
SELECT m.operational_shift,
ROUND(m.shift_revenue,2) AS shift_revenue,
ROUND((m.shift_revenue/ t.grand_total) * 100, 2) AS percentage_contribution
FROM metrics m
CROSS JOIN total_revenue t 
ORDER BY shift_revenue DESC;
-- Answer: Evening shift contributes 62.75% of total revenue, which is more than twice the contribution of the midday shift (25.38%).
-- Although the evening shift does not have the highest sales per transaction (midday shift holds that record), it consistently contributes the most to overall company revenue.
-- This suggests that during the evening, customers tend to buy fewer items per transaction — likely because people shop after work (9-5 jobs) and only purchase a few items, such as food for dinner.
-- This insight could help optimize operational staffing during this specific time period.

-- 3k. Which operational shift contributes most to overall company revenue?
WITH metrics AS(
SELECT SUM(total) AS total_revenue,
CASE
WHEN EXTRACT(HOUR FROM time) BETWEEN 4 AND 9 THEN 'morning'
WHEN EXTRACT(HOUR FROM time) BETWEEN 10 AND 14 THEN 'midday'
WHEN EXTRACT(HOUR FROM time) BETWEEN 15 AND 21 THEN 'evening'
ELSE 'midnight'
END AS operational_shift
FROM walmart
GROUP BY operational_shift
)
SELECT operational_shift,ROUND(total_revenue, 2) AS total_revenue
FROM metrics 
ORDER BY total_revenue DESC
LIMIT 1;
-- Answer: Evening shift (15-21) contributes the most to overall company revenue, generating 3,795,693 across all branches and cities.


-- 4. Customer & Rating Analysis
-- 4a. Are branches with higher average ratings also more profitable on average?
WITH branch_metrics AS(
SELECT branch, 
ROUND(AVG(rating), 4) AS avg_rating, 
ROUND(AVG(profit_margin * total), 4) AS avg_profit
FROM walmart 
GROUP BY branch
)
SELECT 
ROUND((COUNT(*) * SUM(avg_rating * avg_profit) - SUM(avg_rating) * SUM(avg_profit)) / 
(SQRT(COUNT(*) * SUM(avg_rating * avg_rating) - POW(SUM(avg_rating), 2)) *
SQRT(COUNT(*) * SUM(avg_profit * avg_profit) - POW(SUM(avg_profit), 2))), 4) AS correlation
FROM branch_metrics;
-- Answer: Correlation is weak at 0.3356, indicating no meaningful relationship between branch ratings and profitability.

-- 4b. Which categories receive the most extreme customer ratings (high standard deviation)?
SELECT category,
ROUND(AVG(rating), 2) AS avg_rating,
ROUND(STDDEV(rating), 4) AS rating_stddev,
ROUND(MAX(rating) - MIN(rating), 2) AS rating_range
FROM walmart
GROUP BY category
ORDER BY rating_stddev DESC;
-- Answer: Electronic accessories has the highest standard deviation (1.8) with an average rating of only 5.91, indicating the most polarizing customer opinions.
-- Health and beauty follows with a standard deviation of 1.758 but a higher average rating of 7.0.
-- Fashion accessories has a standard deviation of 1.75 with an average rating of 5.78.

-- 4c. Do customers who buy in larger quantities tend to give higher ratings?
SELECT 
ROUND((COUNT(*) * SUM(quantity * rating) - SUM(quantity) * SUM(rating)) / 
(SQRT(COUNT(*) * SUM(quantity * quantity) - POW(SUM(quantity), 2)) *
SQRT(COUNT(*) * SUM(rating * rating) - POW(SUM(rating), 2))), 4) AS correlation
FROM walmart;
-- Answer: Correlation is very weak (1367), indicating no meaningful relationship

-- 4d. Which payment methods are most associated with higher average transaction totals?
SELECT payment_method, COUNT(*) AS transaction_count,
ROUND(AVG(total), 2) AS avg_transaction_total
FROM walmart
GROUP BY payment_method
ORDER BY avg_transaction_total DESC;
-- Answer: Cash has the highest average transaction total (143.88) but the lowest transaction frequency (9,160 transactions).
-- Credit card has the highest transaction frequency (21,280 transactions) but a lower average transaction total (114.85).
-- This suggests that customers tend to use credit cards more often for everyday, lower-priced purchases, while cash is used for fewer but higher-value transactions.
-- 4e. What proportion of total revenue comes from the top 10% highest-spending customers?
SELECT 
    ROUND((SUM(total_spending) / (SELECT SUM(total) FROM walmart)) * 100, 2) AS percentage_contribution
FROM (SELECT invoice_id, SUM(total) AS total_spending
FROM walmart
GROUP BY invoice_id
ORDER BY total_spending DESC
LIMIT 997
) AS top_10;
-- Answer: Top 10% of highest-spending customers contribute 30.23% of total revenue

-- 4f. Which variable (unit price, quantity, or rating) has the strongest correlation with profit margin?
SELECT 
ROUND((COUNT(*) * SUM(unit_price * profit_margin) - SUM(unit_price) * SUM(profit_margin)) / 
(SQRT(COUNT(*) * SUM(unit_price * unit_price) - POW(SUM(unit_price), 2)) *
SQRT(COUNT(*) * SUM(profit_margin * profit_margin) - POW(SUM(profit_margin), 2))), 4) AS corr_unitprice_margin,
ROUND((COUNT(*) * SUM(quantity * profit_margin) - SUM(quantity) * SUM(profit_margin)) / 
	(SQRT(COUNT(*) * SUM(quantity * quantity) - POW(SUM(quantity), 2)) *
	SQRT(COUNT(*) * SUM(profit_margin * profit_margin) - POW(SUM(profit_margin), 2))), 4) AS corr_quantity_margin,
ROUND((COUNT(*) * SUM(rating * profit_margin) - SUM(rating) * SUM(profit_margin)) / 
	(SQRT(COUNT(*) * SUM(rating * rating) - POW(SUM(rating), 2)) *
	SQRT(COUNT(*) * SUM(profit_margin * profit_margin) - POW(SUM(profit_margin), 2))), 4) AS corr_rating_margin
FROM walmart;
-- Answer: Rating has the highest correlation with profit margin (0.0659), followed by quantity (0.0021), and unit price (-0.0111).
-- This indicates that Walmart's profit margin is not driven by product price, but by how frequently customers buy specific products with higher profit margins.
-- Ratings may not significantly impact profit growth because Walmart is easily accessible in many areas, so customers don't heavily rely on ratings when making purchases.

-- 4g. Are there categories that generate high sales but have below-average profit margins?
WITH category_sales AS(
SELECT category, SUM(total) AS category_total, 
	AVG(profit_margin) AS avg_cat_margin
FROM walmart 
GROUP BY category
),
overall_profit_margin AS(
SELECT AVG(profit_margin) AS avg_overall_margin
FROM walmart
)
SELECT c.category,
CASE 
	WHEN c.category_total > (SELECT AVG(total) FROM walmart) 
	AND c.avg_cat_margin < o.avg_overall_margin THEN 'Yes'
	ELSE 'No'
    END AS high_sales_low_margin
FROM category_sales c
CROSS JOIN overall_profit_margin o
ORDER BY c.category_total DESC;
-- Answer: The categories with high sales but below-average profit margins are Fashion accessories, Electronic accessories, and Sports and travel.
-- Customer spending allocation varies significantly across these categories. If the profit margin is not adjusted properly for pricing, customers may not buy the product even after price cuts.

-- 4h. Which branches show a decrease in profit margin despite increased total sales compared to the previous month?
WITH monthly_sales AS(
SELECT branch, SUM(total) AS total_sales, 
	AVG(profit_margin) AS avg_profit_margin, 
	EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month 
FROM walmart
GROUP BY branch, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
),
lag_monthly_sales AS(
SELECT *, 
	LAG(total_sales) OVER(PARTITION BY branch ORDER BY month) AS prev_total_sales, 
	LAG(avg_profit_margin) OVER(PARTITION BY branch ORDER BY month) AS prev_avg_profit_margin
FROM monthly_sales
),
flag_calculation AS(
SELECT branch,
CASE 
WHEN avg_profit_margin < prev_avg_profit_margin AND total_sales > prev_total_sales THEN 1
ELSE 0
END AS flag
FROM lag_monthly_sales 
WHERE prev_total_sales IS NOT NULL
)
SELECT branch,
CASE WHEN SUM(flag) > 0 THEN 'Yes'
ELSE 'No'
END AS profit_margin_declining
FROM flag_calculation
GROUP BY branch;
-- Answer: The majority of Walmart branches show declining profit margins despite higher sales.
-- Only a few branches (WALM005, WALM008, WALM010, WALM011, WALM012, WALM013, WALM016, WALM021, WALM024, WALM028, WALM033, WALM034, WALM037, WALM039, WALM041, WALM045, WALM052) do not experience this issue.
-- This could happen if these branches sell products with higher profit margins or have better cost management compared to others.

-- 4i. Which category yields the highest profit per unit sold?
WITH metrics AS(
SELECT category, COUNT(invoice_id) AS unit_sold, 
SUM(profit_margin * total) AS cat_total_profit
FROM walmart 
GROUP BY category
)
SELECT 
category, 
ROUND(cat_total_profit / unit_sold, 4) AS profit_per_unit
FROM metrics
ORDER BY profit_per_unit DESC
LIMIT 1;
-- Answer: Sports and travel has the highest profit per unit sold with 124.18 values

-- 4j. Which branches show declining profitability over time even if total revenue increases?
WITH branch_sales AS(
SELECT branch, SUM(profit_margin * total) AS total_profit, 
SUM(total) AS total_revenue, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month
FROM walmart
GROUP BY branch, EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
),
lag_branch_sales AS(
SELECT *, 
LAG(total_profit) OVER(PARTITION BY branch ORDER BY month) AS prev_total_profit, 
LAG(total_revenue) OVER(PARTITION BY branch ORDER BY month) AS prev_total_revenue 
FROM branch_sales 
)
SELECT DISTINCT branch,
CASE 
WHEN total_profit < prev_total_profit AND total_revenue > prev_total_revenue THEN 'Yes'
ELSE 'No'
END AS declining_profitability
FROM lag_branch_sales
WHERE prev_total_profit IS NOT NULL
ORDER BY branch;
-- Answer: WALM051 and WALM096 show declining profitability even when total revenue increases.
-- This suggests that customers are buying more products with lower profit margins. Sales remain high, but the branches are not experiencing healthy profit growth.

-- 4k. Which branch would benefit most from weekend promotions?
WITH weekend_vs_weekday AS(
SELECT branch,
CASE WHEN DAYOFWEEK(STR_TO_DATE(date, '%d/%m/%y')) IN (6,7) THEN 'weekend'
ELSE 'weekday'
END AS day_type,
SUM(total) AS total_sales,
SUM(profit_margin * total) AS total_profit
FROM walmart
GROUP BY branch, 
CASE 
	WHEN DAYOFWEEK(STR_TO_DATE(date, '%d/%m/%y')) IN (6,7) THEN 'weekend'
	ELSE 'weekday'
	END
),
branch_comparison AS(
SELECT branch,
	MAX(CASE WHEN day_type = 'weekend' THEN total_sales END) AS weekend_sales,
	MAX(CASE WHEN day_type = 'weekday' THEN total_sales END) AS weekday_sales,
	MAX(CASE WHEN day_type = 'weekend' THEN total_profit END) AS weekend_profit,
	MAX(CASE WHEN day_type = 'weekday' THEN total_profit END) AS weekday_profit
FROM weekend_vs_weekday
GROUP BY branch
)
SELECT branch,
ROUND(((weekend_profit - weekday_profit) / weekday_profit) * 100, 2) AS weekend_profit_gain_percent,
CASE 
WHEN weekend_profit > weekday_profit THEN 'Yes'
ELSE 'No'
END AS should_promote_weekend
FROM branch_comparison
ORDER BY weekend_profit_gain_percent DESC;
-- Answer: There are no branches with positive profit gains on weekends, indicating that weekend promotions may not be necessary.
-- Profit margins on weekends tend to be lower than weekdays, possibly because customers are still shopping
-- Since customers already have higher buying power on weekends and tend to purchase more expensive items, additional promotions may not be required.

-- 4l. If the company wants to boost profit by 10%, which variable (price, quantity, or margin) should be optimized first?
WITH profit_calculation AS(
SELECT quantity, unit_price, profit_margin, (profit_margin * total) AS profit
FROM walmart
)
SELECT 
    ROUND((COUNT(*) * SUM(quantity * profit) - SUM(quantity) * SUM(profit)) / 
          (SQRT(COUNT(*) * SUM(quantity * quantity) - POW(SUM(quantity), 2)) *
           SQRT(COUNT(*) * SUM(profit * profit) - POW(SUM(profit), 2))), 4) AS quantity_profit_corr,
    ROUND((COUNT(*) * SUM(unit_price * profit) - SUM(unit_price) * SUM(profit)) / 
          (SQRT(COUNT(*) * SUM(unit_price * unit_price) - POW(SUM(unit_price), 2)) *
           SQRT(COUNT(*) * SUM(profit * profit) - POW(SUM(profit), 2))), 4) AS price_profit_corr,
    ROUND((COUNT(*) * SUM(profit_margin * profit) - SUM(profit_margin) * SUM(profit)) / 
          (SQRT(COUNT(*) * SUM(profit_margin * profit_margin) - POW(SUM(profit_margin), 2)) *
           SQRT(COUNT(*) * SUM(profit * profit) - POW(SUM(profit), 2))), 4) AS margin_profit_corr
FROM profit_calculation;
-- Answer: Quantity has the strongest correlation with profit compared to the other two variables. This suggests that selling more units, even of lower-priced items, can compensate for lower profit margins through higher volume.
-- Therefore, increasing sales volume through inventory turnover is the right strategy for Walmart branches and can be applied across all branches.

-- 4m. Which combination of city and category produces the highest profit density (profit per transaction)?
WITH metrics AS(
SELECT category, city, COUNT(invoice_id) AS transactions, SUM(profit_margin * total) AS total_profit
FROM walmart
GROUP BY city, category
)
SELECT city, category, 
    ROUND(total_profit / transactions, 4) AS profit_per_transaction
FROM metrics
ORDER BY profit_per_transaction DESC
LIMIT 5;
-- Answer: McKinney + Sports and travel has the highest profit density, with profit per transaction reaching 458.11

-- 4n. How has the relationship between unit price and rating changed over time?
WITH monthly_corr AS(
SELECT 
EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y')) AS month,
(COUNT(*) * SUM(unit_price * rating) - SUM(unit_price) * SUM(rating)) / 
(SQRT(COUNT(*) * SUM(unit_price * unit_price) - POW(SUM(unit_price), 2)) *
SQRT(COUNT(*) * SUM(rating * rating) - POW(SUM(rating), 2))) AS price_rating_corr
FROM walmart
GROUP BY EXTRACT(MONTH FROM STR_TO_DATE(date, '%d/%m/%y'))
)
SELECT month,ROUND(price_rating_corr, 4) AS correlation,
CASE 
WHEN price_rating_corr > 0.5 THEN 'Strong positive'
WHEN price_rating_corr > 0.2 THEN 'Weak positive'
WHEN price_rating_corr > -0.2 THEN 'No correlation'
WHEN price_rating_corr > -0.5 THEN 'Weak negative'
ELSE 'Strong negative'
END AS correlation_strength
FROM monthly_corr
ORDER BY month;
-- Answer: The correlation fluctuates every month across Walmart stores. There is even negative correlation for a few months, but the values are very small and close to zero. Basically, there is no correlation between unit price and customer rating.
