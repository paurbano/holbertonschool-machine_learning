-- ranks country origins of bands
SELECT origin, SUM(fans) AS nb_fans 
FROM metal_bands 
WHERE origin <> '' 
GROUP BY origin HAVING nb_fans > 1 ORDER BY nb_fans DESC;
