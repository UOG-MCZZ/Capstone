CREATE TABLE CLS1B (
    InitialCBWCertNumber varchar(32),
    CBWCertNumber varchar(32),
    CompanyName varchar(128),
    SurveillanceYearStartDate date,
    SurveillanceYearEndDate date,
    FireLabSamplingMonitoring date,
    SurveillanceStatusMonitoring varchar(64),
    IssueDate date,
    Revision date,
    ExpiryDate date,
    SurveillanceFrequency varchar(16),
    ProductType varchar(128)
    );

INSERT INTO CLS1B VALUES ("CLS1B 12312 Rev. 00", "", "Maggie Ponzi Engineering", "6/5/2024", "7/5/2025", "7/11/2024", "First Surveillance Year", "6/5/2024", "", "7/5/2025", "Annually", "Fire Rated Doors");