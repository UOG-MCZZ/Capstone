CREATE TABLE CLS1B (
    InitialCBWCertNumber varchar(32),
    CBWCertNumber varchar(32) NOT NULL,
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

INSERT INTO CLS1B VALUES ("CLS1B 12312 Rev. 00", "", "Maggie Ponzi Engineering", "6/5/2024", "7/5/2025", "7/11/2024", "First Surveillance Year", "6/5/2024", "", "7/5/2025", "Annually", "Fire Rated Doors");

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

INSERT INTO CLS1B_MEC VALUES ("CLS1B 12312 Rev. 00", "Tommy DIck", "12094", "11/5/2024", "9/5/2024", "5/6/2024", "F", "21-Jun-24 Follow up with Lab")
