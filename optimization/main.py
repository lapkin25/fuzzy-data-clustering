from input_data import *

invest_to_compet = InvestToCompet("invest_to_compet.csv")
compet_t0 = CompetData("data_compet_t0.csv")
expectations = ExpectationsData("data_deviation_expectations.csv")

#activities_expectations = ActivitiesExpectations("expectations.csv")
expectations_to_burnout = ExpectationsToBurnout(expectations)
