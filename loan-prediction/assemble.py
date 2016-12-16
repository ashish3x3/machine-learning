import os
import settings
import pandas as pd
import csv

HEADERS = {
    "Acquisition": [
        "id",
        "channel",
        "seller",
        "interest_rate",
        "balance",
        "loan_term",
        "origination_date",
        "first_payment_date",
        "ltv",
        "cltv",
        "borrower_count",
        "dti",
        "borrower_credit_score",
        "first_time_homebuyer",
        "loan_purpose",
        "property_type",
        "unit_count",
        "occupancy_status",
        "property_state",
        "zip",
        "insurance_percentage",
        "product_type",
        "co_borrower_credit_score"
    ],
    "Performance": [
        "id",
        "reporting_period",
        "servicer_name",
        "interest_rate",
        "balance",
        "loan_age",
        "months_to_maturity",
        "maturity_date",
        "msa",
        "delinquency_status",
        "modification_flag",
        "zero_balance_code",
        "zero_balance_date",
        "last_paid_installment_date",
        "foreclosure_date",
        "disposition_date",
        "foreclosure_costs",
        "property_repair_costs",
        "recovery_costs",
        "misc_costs",
        "associated_taxes_for_property",
        "tax_costs",
        "sale_proceeds",
        "credit_enhancement_proceeds",
        "repurchase_proceeds",
        "other_foreclosure_proceeds",
        "non_interest_bearing_balance",
        "principal_forgiveness_balance",
        "repurchase_proceeds_flag"
    ]
}

SELECT = {
    "Acquisition": HEADERS["Acquisition"],
    "Performance": [
        "id",
        "foreclosure_date"
    ]
}

def concatenate(prefix="Acquisition"):
    files = os.listdir(settings.DATA_DIR)
    full = []
    for f in files:
        # print 'f ',f
        if not f.startswith(prefix):
            print 'f does not startts with prefix ', f
            continue
        print 'f starts with prefix ', f
        # csvFileArray = []
        # # for filename in files:
        # with open(os.path.join(settings.DATA_DIR,f), 'rb') as f:
        #     pamreader = csv.reader(f, delimiter='|')
        #     for row in pamreader:
        #         print ', '.join(row)
        #         csvFileArray.append(row)

        if prefix == 'Performance':
            data = pd.read_csv(os.path.join(settings.DATA_DIR, f), sep="|", header=None, names=HEADERS[prefix],
                               index_col=False,nrows=1266580)
        else:
            data = pd.read_csv(os.path.join(settings.DATA_DIR, f), sep="|", header=None, names=HEADERS[prefix],
                               index_col=False)
        # print 'data ',data.head(5)
        data = data[SELECT[prefix]]
        print 'data ', data.head(5)
        full.append(data)

    full = pd.concat(full, axis=0)

    full.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.txt".format(prefix)), sep="|", header=SELECT[prefix], index=False)

if __name__ == "__main__":
    # concatenate("Acquisition")
    print '##################################'
    concatenate("Performance")
