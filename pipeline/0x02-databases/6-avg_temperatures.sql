-- average temperature
SELECT CITY, AVG(VALUE) AS avg_temp 
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;
