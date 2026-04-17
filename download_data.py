import requests
import time
import pandas as pd


def fetch_monomer_physicochemical_properties(limit_pages=None):
    base_url = "https://dbaasp.org"
    peptides_url = f"{base_url}/peptides"

    headers = {
        "Accept": "application/json"
    }

    properties_by_id = {}

    # Pagination variables
    page = 0
    size = 100  # Number of peptides per page
    has_more = True

    print("Starting download of monomer peptides...")

    while has_more:
        if limit_pages and page >= limit_pages:
            break

        # 1. Get a page of peptide search results
        params = {
            "complexity": "monomer",  # Filter for monomers
            "offset": page*size,
            "limit": size
        }

        print(f"Fetching page {page}...")
        response = requests.get(peptides_url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"Error fetching list: {response.status_code} - {response.text}")
            break

        data = response.json()

        # DBAASP likely uses a standard paginated JSON structure where items are inside 'content'
        items = data.get("data", []) if isinstance(data, dict) else data

        if not items:
            break

        # 2. Iterate through the page's items and fetch the full PeptideView for each
        for item in items:
            peptide_id = item.get("id")
            if not peptide_id:
                continue

            # Fetch the specific PeptideView
            detail_url = f"{base_url}/peptides/{peptide_id}"
            detail_resp = requests.get(detail_url, headers=headers)

            if detail_resp.status_code == 200:
                peptide_view = detail_resp.json()

                # Extract physicoChemicalProperties under PeptideView
                phys_props = peptide_view.get("physicoChemicalProperties")
                if phys_props:
                    properties_by_id[peptide_id] = phys_props

                # TODO extract other data
                # inter/intra chain bonds?

                #targetActivities
                #antibiofilmActivities
                #hemoliticCytotoxicActivities

            # Brief pause to respect API rate limits
            time.sleep(0.05)

        # 3. Check if there are more pages
        if isinstance(data, dict):
            page += 1
        else:
            # If the API just returns a flat list and isn't paginating
            has_more = False

    # process dict
    # 1. Reformat the structure so that each ID maps to a simple dictionary of {name: value}
    reformatted_data = {
        peptide_id: {prop['name']: prop['value'] for prop in properties_list}
        for peptide_id, properties_list in properties_by_id.items()
    }

    # 2. Create the DataFrame, using the dictionary keys (the original IDs) as the index
    df = pd.DataFrame.from_dict(reformatted_data, orient='index')

    # The API returns all values as strings (e.g., '1.97', '5.00').
    # This line safely converts columns containing numbers into actual float/int data types.
    num_cols = df.columns[1:]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    return df


if __name__ == "__main__":
    # Remove `limit_pages=1` to scrape the entire database (this will take time!)
    # We are using limit_pages=1 here just to test the first 100.
    monomer_data = fetch_monomer_physicochemical_properties(limit_pages=40)

    print(f"\nSuccessfully fetched properties for {len(monomer_data)} peptides.")

    monomer_data.to_csv('./subset_chem_phys_features.txt',sep='\t')

    # Print an example of the first fetched peptide
    if monomer_data is not None:
        sample_id = list(monomer_data.keys())[0]
        print(f"\nSample Data for Peptide ID {sample_id}:")
        print(monomer_data[sample_id])