-- lists all bands with Glam rock as their main style
SELECT band_name, IFNULL(split, 2020) - formed AS lifespan
FROM metal_bands 
WHERE style LIKE '%Glam rock%' ORDER BY 2 DESC;
