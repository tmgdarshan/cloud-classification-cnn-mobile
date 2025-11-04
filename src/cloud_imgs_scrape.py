import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

CHROMEDRIVER_PATH = "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/chromedriver-linux64/chromedriver"  # <<< Update this!

cloud_classes = {
    "cirrus": "cirrus clouds",
    "cumulus": "cumulus clouds",
    "cb": "cumulonimbus clouds",
}


def download_image(url, save_folder, count):
    try:
        img_data = requests.get(url, timeout=10).content
        filename = os.path.join(save_folder, f"img_{count}.jpg")
        with open(filename, "wb") as img_file:
            img_file.write(img_data)
    except Exception as e:
        print(f"Error downloading {url}: {e}")


service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service)

for label, query in cloud_classes.items():
    save_folder = f"data/raw/{label}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"\nScraping images for '{query}'...")

    # Open Google Images
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
    driver.get(url)
    time.sleep(4)

    # Try to accept consent/dialog
    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            txt = btn.text.lower()
            if "accept" in txt or "agree" in txt or "consent" in txt:
                btn.click()
                print("Accepted cookies/privacy dialog.")
                time.sleep(2)
                break
    except Exception as e:
        print("Consent dialog click not needed/found.")

    # Scroll several times to load images
    for _ in range(4):
        driver.execute_script("window.scrollBy(0, 1800)")
        time.sleep(2)

    time.sleep(3)

    images = driver.find_elements(By.TAG_NAME, "img")
    print(f"Found {len(images)} <img> tags.")

    count = 0
    for i, img in enumerate(images):
        src_url = img.get_attribute("src")
        data_src_url = img.get_attribute("data-src")
        print(f"Image {i}: src={src_url}, data-src={data_src_url}")

        url = src_url if src_url and src_url.startswith("http") else None
        if not url and data_src_url and data_src_url.startswith("http"):
            url = data_src_url
        if url and url.startswith("http"):
            download_image(url, save_folder, count)
            count += 1
            print(f"Downloaded image {count} for {label}")
        if count >= 30:
            break
    print(f"Downloaded {count} images for '{label}'.\n")

driver.quit()
print("Scraping complete.")
