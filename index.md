---
layout: home
title: "My Tutorials"
---

# Welcome to TradeMamba 🎉

This is my GitHub Pages site. Here are some useful links:

## ✍️ My Blog Posts
Check out my latest Medium articles here: [Medium Posts →](/medium/)


## 🔗 Subpages
- [AMD Tutorials]({{ "/amdtutorials/" | relative_url }})
- [Finance Notes]({{ "/finance/" | relative_url }})
- [About Me]({{ "/about/" | relative_url }})

[View my notebook as a blog post]({{ "/blog/bible-rag-actual/" | relative_url }})
or
[Open the notebook on nbviewer](https://nbviewer.org/github/TradeMamba/amdtutorials/blob/main/finance/Bible%20Rag%20Actual.ipynb)

## 📝 Latest Posts
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      — {{ post.date | date: "%b %-d, %Y" }}
    </li>
  {% endfor %}
</ul>

## 📓 Jupyter Notebooks
- [Intro Notebook (nbviewer)](https://nbviewer.org/github/TradeMamba/amdtutorials/blob/main/notebooks/intro.ipynb)
- [Options Trading Notebook (GitHub view)](https://github.com/TradeMamba/amdtutorials/blob/main/notebooks/options_trading.ipynb)
- [Run on Binder](https://mybinder.org/v2/gh/TradeMamba/amdtutorials/HEAD?labpath=notebooks/intro.ipynb)

## 🌍 External Links
- [YouTube Channel](https://youtube.com)
- [Medium Articles](https://medium.com/@TradeMamba)
- [GitHub Profile](https://github.com/TradeMamba)

## 📬 Contact
- Email: [yourname@example.com](mailto:yourname@example.com)
- Twitter: [@TradeMamba](https://twitter.com/TradeMamba)
