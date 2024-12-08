from pymongo import MongoClient, UpdateOne
import re
from config import *

# 1. Connect to MongoDB
client = MongoClient(MONGO_HOST, MONGO_PORT)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]
new_collection_name = 'new_' + MONGO_COLLECTION  # Name of the new collection
new_collection = db[new_collection_name]

# 2. Function to copy documents to a new collection
def copy_collection():
    """
    Copies all documents from the original collection to a new collection.
    The new collection will be named 'new_<original_collection_name>'.
    """
    if new_collection.count_documents({}) > 0:
        print(f"New collection '{new_collection_name}' already exists and is not empty. Skipping copy.")
        return
    
    print(f"Copying documents from '{MONGO_COLLECTION}' to '{new_collection_name}'...")
    documents = list(collection.find())
    if documents:
        new_collection.insert_many(documents)
    print(f"Copied {len(documents)} documents to '{new_collection_name}'.")


# 3. Function to normalize URL
def normalize_url(url):
    """
    Normalizes the URL by keeping only the part up to preview_entity.php?catoid=XX&ent_oid=XX&returnto=XX.
    Removes everything after 'returnto=XX'.
    """
    pattern = r"(https:\/\/catalog\.fullerton\.edu\/preview_entity\.php\?[^&]+&[^&]+&returnto=\d+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    return url  # Return the original URL if it doesn't match the pattern


# 4. Filter and clean URLs in the new collection
def clean_urls_in_collection():
    """
    Filters, cleans, and updates the URL field for all documents in the new MongoDB collection.
    If the URL contains 'content.php', 'index.php', or any extraneous parts after 'returnto=XXXX',
    it will be truncated.
    """
    processed_count = 0
    bulk_updates = []

    print(f"Cleaning URLs in '{new_collection_name}, {new_collection.count_documents({})}'...")

    cursor = new_collection.find({}, {"_id": 1, "url": 1})  # Only retrieve the _id and url to optimize performance

    for doc in cursor:
        original_url = doc.get("url", "")
        if not original_url:
            continue

        # Normalize the URL
        cleaned_url = normalize_url(original_url)
        
        if cleaned_url != original_url:
            print(f"Cleaning URL: {original_url} -> {cleaned_url}")
            bulk_updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},  # Make sure _id is referenced correctly
                    {"$set": {"url": cleaned_url}}
                )
            )
            processed_count += 1

    if bulk_updates:
        try:
            result = new_collection.bulk_write(bulk_updates)
            print(f"Updated {result.modified_count} documents with cleaned URLs.")
        except Exception as e:
            print(f"Bulk write failed: {e}")

    print(f"Processed {processed_count} documents and cleaned URLs.")

def remove_unwanted_domains():
    unwanted_domains = [
        "titansgive.fullerton",
        "giveto.fullerton",
        "give.fullerton"
    ]
    delete_query = {
        "url": {
            "$regex": f"({'|'.join(unwanted_domains)})"
        }
    }
    try:
        delete_result = new_collection.delete_many(delete_query)
        print(f"Deleted {delete_result.deleted_count} documents with unwanted URLs from '{new_collection_name}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")

def count_domains():
    """
    Counts the occurrences of each domain in the URL field.
    The domain is extracted from the URL, and all occurrences are aggregated.
    """
    print(f"Counting domain occurrences in '{new_collection_name}'...")

    # Extract the domain from the URL using regex and count occurrences
    cursor = new_collection.find({}, {"url": 1})
    domain_counts = {}

    for doc in cursor:
        url = doc.get("url", "")
        if not url:
            continue

        # Extract domain using regex (e.g., catalog.fullerton.edu, titansgive.fullerton.edu)
        match = re.match(r"https?://([^/]+)", url)
        if match:
            domain = match.group(1)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Sort by count (descending) and print results
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nDomain counts (most common first):")
    for domain, count in sorted_domains:
        print(f"{domain}: {count} occurrences")

    return sorted_domains

# 5. Filter and remove unnecessary documents from the new collection
def remove_unwanted_urls():
    """
    Removes documents where the URL contains 'content.php', 'index.php', or 
    extra data beyond 'returnto=XXX' in the preview_entity.php link.
    """
    delete_query = {
        "url": {
            "$regex": r"(content\.php|index\.php)"
        }
    }
    
    try:
        delete_result = new_collection.delete_many(delete_query)
        print(f"Deleted {delete_result.deleted_count} documents with unwanted URLs from '{new_collection_name}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")


non_matching_regex = r"https:\/\/catalog\.fullerton\.edu\/(?!content\.php|preview_program\.php)"
def count_non_matchin_urls():

    query = {
        "url": {
            "$regex": non_matching_regex
        }
    }
    
    try:
        count = new_collection.count_documents(query)
        print(f"Found {count} documents with non-matching URLs in '{new_collection_name}'.")
    except Exception as e:
        print(f"Error during count_documents: {e}")
def remove_non_matching_urls():
    """
    Remove documents where the URL for catalog.fullerton.edu does not match:
    - https://catalog.fullerton.edu/content.php?
    - https://catalog.fullerton.edu/preview_program.php?
    """
    query = {
        "url": {
            "$regex": non_matching_regex
        }
    }
    
    try:
        delete_result = new_collection.delete_many(query)
        print(f"Deleted {delete_result.deleted_count} documents with non-matching URLs from '{new_collection_name}'.")
    except Exception as e:
        print(f"Error during delete_many: {e}")

# 6. Run the copy, cleaning, and removal process
if __name__ == "__main__":

    print("\nStep 0: Counting domains")
    #count_domains()

    print("Step 1: Copy collection to new collection...")
    #copy_collection()
    
    print("\nStep 2: Cleaning URLs in new collection...")
    clean_urls_in_collection()
    
    print("\nStep 3: Removing documents with unnecessary URLs...")
    remove_unwanted_urls()

    print("\nStep 5: Count Old urls")
    count_non_matchin_urls()

    print("\nStep 6: Remove Old urls")
    remove_non_matching_urls()

    print("\nStep 7: Remove unwanted domains")
    #remove_unwanted_domains()

    print("\nStep 10: Counting domains")
    #count_domains()

