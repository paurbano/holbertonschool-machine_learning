-- adds a new correction for a student
DELIMITER //
CREATE PROCEDURE AddBonus (IN user_id INT, IN project_name varchar(255), IN score INT)
BEGIN
	IF NOT EXISTS (SELECT 1 FROM projects WHERE name = project_name) THEN
		INSERT INTO  projects (name) VALUES (project_name);
        SET @project_id = LAST_INSERT_ID();
	ELSE
			SELECT @project_id = id FROM projects WHERE name = project_name;
    END IF;
	INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, @project_id, score);
END;
DELIMITER ;
