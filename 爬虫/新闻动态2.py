import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime

head={
"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0"
}

urls = []

url="https://youth.cumt.edu.cn/index/xwdt.htm"
resp=requests.get(url,headers=head)
content_page = BeautifulSoup(resp.text, "html.parser")
u_list = content_page.find("div", class_="list_news",style="min-height:500px;").findAll("a")
for u in u_list:
    href = u.get("href")
    if href:
        full_url=href.replace("../../", "")
        full_url="https://youth.cumt.edu.cn/"+full_url
        urls.append(full_url)

for i in range(68, 0, -1):
    url=f"https://youth.cumt.edu.cn/index/xwdt/{i}.htm"
    resp=requests.get(url,headers=head)
    content_page = BeautifulSoup(resp.text, "html.parser")
    u_list = content_page.find("div", class_="list_news",style="min-height:500px;").findAll("a")
    for u in u_list:
        href = u.get("href")
        if href:
            full_url=href.replace("../../", "")
            full_url="https://youth.cumt.edu.cn/"+full_url
            urls.append(full_url)

news=[]
fetch_time = datetime.now().strftime('%Y-%m-%d')

for u in urls:
    url=u
    resp=requests.get(url)
    resp.encoding = "utf-8"

    page = BeautifulSoup(resp.text, "html.parser")
    # page_article=page.find("div",class_="news_hd f16 tc")
    page_article=page.find("div",class_="news_hd f16 tc")
    article_data = ""
    if page_article:
        page_title=page_article.find("h3")
        title=page_title.get_text(strip=True)
        editor=page_article.find("p").find_all("span")[0].get_text(strip=True)
        content =title+"\n"+editor
        article_content = page.find("div",class_="news_bd")
        article_paragraphs = article_content.findAll('p')
        for p in article_paragraphs:
            paragraphs=p.get_text(strip=True)
            content+="\n"+ paragraphs
        date_info = page_article.find("p")
        date = ""
        if date_info:
            date_span = date_info.find_all("span")[-1]  # 获取最后一个<span>，通常是时间
            date = date_span.get_text(strip=True).replace("时间：", "")
        article_data = {
            "url": url,
            "content": content,
            "date": date
        }
    news.append(article_data)

output_data = {
    "data": news,
    "fetch_time": fetch_time
}
with open('news.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print("over!!")
resp.close()