--  lists all shows that have at least one genre linked.
SELECT tv_shows.title,tv_show_genres.genre_id FROM tv_shows
INNER JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
INNER JOIN tv_genres ON tv_genres.id = tv_show_genres.genre_id
ORDER BY 1,2;
