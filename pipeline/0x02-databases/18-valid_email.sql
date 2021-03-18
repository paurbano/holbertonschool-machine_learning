-- resets the attribute valid_email
DELIMITER //
CREATE TRIGGER reset_email BEFORE UPDATE ON users
	FOR EACH ROW
	BEGIN
		IF STRCMP(OLD.email, NEW.email) <>0 THEN
			UPDATE users SET valid_email = 0;
        END IF;
	END;
DELIMITER;
