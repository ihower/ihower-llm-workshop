static 目錄下的 html/js/css 檔案都是用 Vibe coding 做的，以下是 prompt 參考:

## completion.html

請製作 HTML 頁面，左欄是一個 textarea 的 form，按下 Submit 後
用 javascript 送出 GET 到 /api/v1/completion_stream 帶有參數 query
回傳是用 SSE 回來，請將串流的資料，顯示在畫面右欄。當收到 [DONE] 時，可以結束 SSE 串流。
然後清空左邊的 form 資料。如果再次 Submit，右欄需要先清空
畫面用 bootstrap 裝飾一下(用 CDN 載入 css)


## structured_output.html

請製作 HTML 頁面，左欄是一個 textarea 的 form，按下 Submit 後
用 javascript 送出 GET 到 /api/v1/completion_json_stream 帶有參數 query
回傳是用 SSE 傳流回來，會是合法的 json result

請將收到的 result["content"] 累加在右欄的 content div 裡面
請將收到的 result["following_questions"]  替換右欄的 following_questions div 整塊，這是個 array，請用 ul li 顯示
當收到 { "message": "DONE" } 時，可以結束 SSE 串流。
最後清空左邊的 form 資料。

如果再次 Submit，右欄需要先清空
畫面用 bootstrap 裝飾一下(用 CDN 載入 css)

