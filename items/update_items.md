# Update `items`

Navigate to `https://prts.wiki/w/ITEM?filter=AvwAAQ` in the browser's address bar to go to PRTS, or run the following javascript in
the browser's console:

```javascript
// Go to PRTS
window.location = "https://prts.wiki/w/ITEM?filter=AvwAAQ"
```

Run the following javascript to download icons and data of the items from PRTS: 

```javascript
// Download Data from PRTS
const a = document.createElement('a');
document.body.appendChild(a);
a.style.display = 'none';

async function download(href, download) {
    a.href = href;
    a.download = download;
    a.click();
    console.log(`download... ${a.download}: ${a.href}`);
    await new Promise((r) => setTimeout(r, 500))
}

for (const it of filterResult) {
    delete it.description;
    delete it.usage;
    delete it.category;
    delete it.obtain_approach;
    delete it.rarity;

    let file = it.file;
    let file_url = file.substring(0, file.lastIndexOf("/")).replace("thumb/", "");
    // Download Items Image in PNG
    await download(file_url, `${it.id}.png`)

    delete it.file;
}

// Download Items Data in JSON
await download(`data:text/plain;charset=utf-8,${JSON.stringify(filterResult)}`, "items.json");
```

Note that the javascript has set a downloading interval for 1000 ms, to protect PRTS's server from crashing. (It spends ~100 seconds to download all ~4 MiB files)
