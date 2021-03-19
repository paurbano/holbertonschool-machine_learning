-- view
CREATE OR REPLACE VIEW need_meeting AS
SELECT name FROM students
WHERE score < 80
AND last_meeting IS NULL OR 
	(MONTH(CURDATE()) - MONTH(last_meeting)) > 1;
