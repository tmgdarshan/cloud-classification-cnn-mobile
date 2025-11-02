from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def main():
    opts = Options()
    opts.add_argument("--headless")
    driver = webdriver.Chrome(options=opts)
    driver.get("https://www.scrapingbee.com/")
    print(driver.title)

    links = driver.find_elements(By.CSS_SELECTOR, ".navbar-wrap nav a")
    if links:
        print(f"Found {len(links)} nav links:")
        for link in links:
            print("-", link.text.strip())

    driver.quit()

if __name__ == "__main__":
    main()