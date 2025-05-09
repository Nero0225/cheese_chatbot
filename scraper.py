import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
import json

ORIGIN = "https://shop.kimelo.com"
URL = "https://shop.kimelo.com/department/cheese/3365"

def scrape_cheese_data(url):
    """Fetches the page content and extracts cheese data."""
    page_content = requests.get(url)

    soup = BeautifulSoup(page_content.text, 'html.parser')

    products = []

    product_elements = soup.find_all("a", class_="chakra-card group css-5pmr4x")

    print("product_elements", len(product_elements))

    for item in product_elements:
        more_url = item.get('href')
        image_url = unquote(item.find('img')['src'].replace("/_next/image?url=", "").replace("&w=3840&q=50", ""))
        name = item.find('p', class_="chakra-text css-pbtft").text
        price = item.find('b', class_="chakra-text css-1vhzs63")
        company_name = item.find('p', class_="chakra-text css-w6ttxb").text
        per_price = item.find('span', class_="chakra-badge css-ff7g47")
        status = item.find('span', class_="chakra-badge css-qrs5r8")
        if status:
            status = status.text
        else:
            status = "exist"

        if price:
            price = price.text
        else:
            price = "N/A"

        if not per_price:
            per_price = "N/A"
        else:
            per_price = per_price.text

        SKU = "N/A"
        UPC = "N/A"
        category = "N/A"
        attempt = 0

        case_status = "N/A"
        print("more_url", more_url)
        if more_url:
            while attempt < 3:
                try:
                    more_content = requests.get(f"{ORIGIN}{more_url}")
                    more_soup = BeautifulSoup(more_content.text, 'html.parser')
                    categories = more_soup.find_all("a", class_="chakra-link chakra-breadcrumb__link css-1vtk5s8")
                    if categories[1]:
                        category = categories[1].text
                    else:
                        category = "N/A"

                    detail_div = more_soup.find_all("div", class_="css-ahthbn")
                    print("detail_div", len(detail_div))
                    description_SKU_and_UPC = detail_div[-1].find_all("p", class_="chakra-text css-0")

                    related_cheese = more_soup.find_all('div', class_="css-1ydflst")
                    relates = []
                    if len(related_cheese) > 0:
                        related_names = related_cheese[-1].find_all('a')
                        for item in related_names:
                            related_item_text = item.find('p', class_="chakra-text css-pbtft").text
                            relates.append({"name": related_item_text, "this_url": f"{ORIGIN}{item.get('href')}"})
                        print("related", relates)

                    description = "N/A"
                    SKU = "N/A"
                    UPC = "N/A"
                    for item in description_SKU_and_UPC:
                        if item.text[:13] == "Description: ":
                            description = item.text[13:]
                        elif item.text[:5] == "SKU: ":
                            SKU = item.find("b", class_="chakra-text css-0").text
                        elif item.text[:5] == "UPC: ":
                            UPC = item.find("b", class_="chakra-text css-0").text

                    small_images = [unquote(image.find('img')['src'].replace("/_next/image?url=", "").replace("&w=3840&q=75", "")) for image in more_soup.find_all('button', class_="chakra-tabs__tab border css-2jmkdc")]

                    table = more_soup.find_all('table')[-1]

                    headers = [th.text for th in table.find_all('th')]
                    if "Each" in headers:
                        case_status = "each"
                    elif "Case" in headers:
                        case_status = "case"
                    else:
                        case_status = "N/A"

                    rows = table.find_all('tr')
                    if case_status == "case":
                        case_number = rows[1].find_all('td')[0].text
                        case_size = rows[2].find_all('td')[0].text
                        case_weight = rows[3].find_all('td')[0].text

                        each_number = rows[1].find_all('td')[1].text
                        each_size = rows[2].find_all('td')[1].text
                        each_weight = rows[3].find_all('td')[1].text
                    elif case_status == "each":
                        case_number = "N/A"
                        case_size = "N/A"
                        case_weight = "N/A"

                        each_number = rows[1].find_all('td')[0].text
                        each_size = rows[2].find_all('td')[0].text
                        each_weight = rows[3].find_all('td')[0].text
                    else:
                        case_number = "N/A"
                        case_size = "N/A"
                        case_weight = "N/A"

                        each_number = "N/A"
                        each_size = "N/A"
                        each_weight = "N/A"

                    warning_text = more_soup.find_all('p', class_="chakra-text css-dw5ttn")[-1].text[9:]
                    break
                except:
                    attempt += 1
                    time.sleep(1)
                    print("add attempt")
                    if attempt == 3:
                        print("expected", more_url)
        products.append({
            "name": name,
            "price": price,
            "category": category,
            "brand": company_name,
            "image_url": image_url,
            "per_price": per_price,
            "status": status,
            "SKU": SKU,
            "UPC": UPC,
            "warning_text": warning_text,
            "small_images": small_images,
            "case_status": case_status,
            "description": description,
            "more_url": f"{ORIGIN}{more_url}",
            "related_items": relates,
            "info": {
                "case_number": case_number,
                "case_size": case_size,
                "case_weight": case_weight,
                "each_number": each_number,
                "each_size": each_size,
                "each_weight": each_weight,
            }
        })

    return products

def save_data(data, filename="scraped_cheese_data.json"):
    """Saves the scraped data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")

def main():
    """Main function to run the scraper."""
    print(f"Scraping data from {URL}...")
    all_products = []

    current_url = URL
    flag = True
    pn = 1

    while flag:
        current_url = f"{URL}?page={pn}"
        page_products = scrape_cheese_data(current_url)
        if len(page_products) == 0:
            flag = False
            break
        else:
            all_products.extend(page_products)
            print(f"Scraped {len(page_products)} products from {current_url}")
        pn += 1


    if all_products:
        save_data(all_products)
        print("Save scraping data successfully.")
    else:
        print("No data scraped.")

if __name__ == "__main__":
    main()