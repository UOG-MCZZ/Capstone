CREATE TABLE CLS1B (
    InitialCBWCertNumber varchar(32),
    CBWCertNumber varchar(32),
    CompanyName varchar(128) NOT NULL,
    SurveillanceYearStartDate date,
    SurveillanceYearEndDate date,
    FireLabSamplingMonitoring date,
    SurveillanceStatusMonitoring varchar(64),
    IssueDate date,
    Revision date,
    ExpiryDate date,
    SurveillanceFrequency varchar(16),
    ProductType varchar(128),
    PRIMARY KEY (InitialCBWCertNumber)
);

-- INSERT INTO CLS1B VALUES ('CLS1B 12312 Rev. 00', '', 'Maggie Ponzi Engineering', '2024-05-06', '2025-05-07', '2024-11-7', 'First Surveillance Year', '2024-05-06', '', '2025-05-07', 'Annually', 'Fire Rated Doors');

CREATE TABLE CLS1B_MEC (
    InitialCBWCertNumber varchar(32),
    OIC varchar(32),
    SurveillanceSN varchar(16),
    SurveilanceDate date,
    SampleReceivedDate date,
    TestCompletionDate date,
    TestResult char,
    Remarks varchar(256),
    PRIMARY KEY (SurveillanceSN),
    FOREIGN KEY (InitialCBWCertNumber) REFERENCES CLS1B(InitialCBWCertNumber)
);
-- INSERT INTO CLS1B_MEC VALUES ('CLS1B 12312 Rev. 00', 'Tommy DIck', '12094', '2024-05-11', '2024-05-09', '2024-06-05', 'F', '21-Jun-24 Follow up with Lab');

CREATE TRIGGER update_CLS1B_MEC AFTER UPDATE ON CLS1B_MEC
FOR EACH ROW 
	update CLS1B AS cert set cert.SurveillanceStatusMonitoring = 'Surveillance Completed' WHERE 
		new.InitialCBWCertNumber = cert.InitialCBWCertNumber AND 
        NEW.TestCompletionDate < cert.SurveillanceYearEndDate 
        ;

CREATE TRIGGER insert_CLS1B_MEC AFTER INSERT ON CLS1B_MEC
FOR EACH ROW 
	update CLS1B AS cert set cert.SurveillanceStatusMonitoring = 'Surveillance Completed' WHERE 
		new.InitialCBWCertNumber = cert.InitialCBWCertNumber AND 
        NEW.TestCompletionDate < cert.SurveillanceYearEndDate 
        ;

CREATE EVENT dailystatusupdate ON SCHEDULE EVERY 1 DAY DO UPDATE CLS1B cert LEFT JOIN CLS1B_MEC mec 
	ON mec.SurveillanceSN = (SELECT mec2.SurveillanceSN FROM CLS1B_MEC mec2 WHERE mec2.InitialCBWCertNumber = cert.InitialCBWCertNumber ORDER BY mec2.SurveilanceDate DESC LIMIT 1)
    SET cert.SurveillanceStatusMonitoring = CASE 
		WHEN cert.SurveillanceYearStartDate < mec.TestCompletionDate < cert.SurveillanceYearEndDate THEN 'Surveillance Completed'
        WHEN cert.SurveillanceYearEndDate < current_date() THEN 'Surveillance Expired'
        WHEN DATEDIFF(CURRENT_DATE(), cert.SurveillanceYearEndDate) < 90 THEN 'Surveillance Expiring'
		WHEN cert.Revision IS NULL THEN 'First Surveillance Peroid'
        ELSE 'Follow-up Surveillance' END
	WHERE cert.SurveillanceStatusMonitoring != 'Surveillance Completed' AND cert.SurveillanceStatusMonitoring != 'Surveillance Expired';

CREATE TABLE FormColumnConverter (
    TableName varchar(64),
    FormFieldName varchar(64),
    ColumnName varchar(64),
    PRIMARY KEY (TableName, FormFieldName)
);