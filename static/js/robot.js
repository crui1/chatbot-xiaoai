$(function () {
	// 点击发送按钮
	$('#btn-send').on('click', function (e) {
		var t = $(this).siblings().val().trim()
		if (!t) return $(this).siblings().val('')

		// 添加输入信息到页面并且滚动到当前信息
		addMessage(t, true)
		$(this).siblings().val('')
		// 请求回复并且添加到页面
		getMsg(t)
	})

	// 输入框按下enter
	var lastKey = 0
	$('#input').on('keyup', function (e) {

		// 判断 按住ctl+enter
		if (!lastKey) {
			lastKey = e.keyCode
			setTimeout(function () {
				lastKey = 0
			}, 300)
		}

		if ((e.keyCode == 13 && lastKey == 17) || (e.keyCode == 17 && lastKey == 13)) {
			var t = $(this).val().trim()
			if (!t) return $(this).val('')

			// 添加输入信息到页面并且滚动到当前信息
			addMessage(t, true)
			$(this).val('')

			// 请求回复并且添加到页面
			getMsg(t)
		}
	})

	function toBottom(div) {
		var bh = $('.chatBox').outerHeight()
		var offsetBottom = div.position().top + div.outerHeight() + $('.chatBox').scrollTop()

		if (offsetBottom > bh) {
			var target = offsetBottom - bh
			$('.chatBox').scrollTop(target)
		}
	}

	function addMessage(text, me = false) {
		if (me) {
			var div = $('<div class="me"><div class="text">' + text +
				'</div><div class="icon"><img src="./static/img/_10.jpg" class="rimg"></div></div>')
		} else {
			var div = $(
				'<div class="he"><div class="icon"><img src="./static/img/_15.jpg" alt="" class="rimg" /></div><div class="text">' +
				text + '</div></div>')
		}
		$('.chatBox').append(div)
		toBottom(div)

	}

	function getMsg(text) {
		let msg = {
			"message": text
		}
		$.ajax({
			method: 'post',
			url: 'http://localhost:8888/api/chat',
			data: JSON.stringify(msg),
			contentType: "application/json;charset=UTF-8",
			success: function (res) {
				// console.log(res)
				if (res.code === 0) {
					// 接受聊天信息
					var msg = res.info.text
					// console.log(msg)
					addMessage(msg)
					// 调用getVoice函数，把文本转化为语音
					// 播放提示音
					$("#audio").prop('src', '/static/audio/tweet.wav')
				}
			}
		})
	}
})
