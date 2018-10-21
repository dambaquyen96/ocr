let model;

const modelURL = location.protocol+'//'+location.hostname+(location.port ? ':'+location.port: '')+'/model';

const predictButton = document.getElementById("predict");
const fileInput = document.getElementById('file');
const total_width = document.getElementById('total_width');
const total_height = document.getElementById('total_height');
const width = document.getElementById('width');
const height = document.getElementById('height');
const x = document.getElementById('x');
const y = document.getElementById('y');
const str_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'あ', 'い', 'う', 'え', 'お', 'か', 'が', 'き', 'ぎ', 'く', 'ぐ', 'け', 'げ', 'こ', 'ご', 'さ', 'ざ', 'し', 'じ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ', 'た', 'だ', 'ち', 'ぢ', 'つ', 'づ', 'て', 'で', 'と', 'ど', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ば', 'ぱ', 'ひ', 'び', 'ぴ', 'ふ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ', 'ま', 'み', 'む', 'め', 'も', 'ゃ', 'や', 'ゅ', 'ゆ', 'ょ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん', 'ァ', 'ア', 'ィ', 'イ', 'ゥ', 'ウ', 'ェ', 'エ', 'ォ', 'オ', 'カ', 'ガ', 'キ', 'ギ', 'ク', 'グ', 'ケ', 'ゲ', 'コ', 'ゴ', 'サ', 'ザ', 'シ', 'ジ', 'ス', 'ズ', 'セ', 'ゼ', 'ソ', 'ゾ', 'タ', 'ダ', 'チ', 'ヂ', 'ツ', 'ヅ', 'テ', 'デ', 'ト', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'バ', 'パ', 'ヒ', 'ビ', 'ピ', 'フ', 'ブ', 'プ', 'ヘ', 'ベ', 'ペ', 'ホ', 'ボ', 'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ャ', 'ヤ', 'ュ', 'ユ', 'ョ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン', 'ヴ', '一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '且', '世', '丘', '丙', '丞', '両', '並', '中', '串', '丸', '丹', '主', '丼', '乃', '久', '之', '乎', '乏', '乗', '乘', '乙', '九', '乞', '也', '乱', '乳', '乾', '亀', '了', '予', '争', '事', '二', '云', '互', '五', '井', '亘', '亙', '些', '亜', '亞', '亡', '交', '亥', '亦', '亨', '享', '京', '亭', '亮', '人', '仁', '今', '介', '仏', '仔', '仕', '他', '付', '仙', '代', '令', '以', '仮', '仰', '仲', '件', '任', '企', '伊', '伍', '伎', '伏', '伐', '休', '会', '伝', '伯', '伴', '伶', '伸', '伺', '似', '伽', '佃', '但', '位', '低', '住', '佐', '佑', '体', '何', '余', '佛', '作', '佳', '併', '使', '侃', '來', '例', '侍', '侑', '供', '依', '価', '侮', '侯', '侵', '侶', '便', '係', '促', '俄', '俊', '俐', '俗', '保', '信', '俣', '修', '俳', '俵', '俸', '俺', '倉', '個', '倍', '倒', '倖', '候', '借', '倣', '値', '倦', '倫', '倭', '倶', '倹', '偉', '偏', '停', '健', '偲', '側', '偵', '偶', '偽', '傍', '傑', '傘', '備', '催', '傭', '傲', '傳', '債', '傷', '傾', '僅', '働', '像', '僑', '僕', '僚', '僞', '僧', '價', '僻', '儀', '億', '儉', '儒', '償', '優', '儲', '允', '元', '兄', '充', '兆', '先', '光', '克', '免', '兎', '児', '兒', '党', '兜', '入', '全', '八', '公', '六', '共', '兵', '其', '具', '典', '兼', '内', '円', '冊', '再', '冒', '冗', '写', '冠', '冥', '冨', '冬', '冴', '冶', '冷', '凄', '准', '凉', '凌', '凍', '凛', '凜', '凝', '凡', '処', '凧', '凪', '凰', '凱', '凶', '凸', '凹', '出', '函', '刀', '刃', '分', '切', '刈', '刊', '刑', '列', '初', '判', '別', '利', '到', '制', '刷', '券', '刹', '刺', '刻', '剃', '則', '削', '前', '剖', '剛', '剣', '剤', '剥', '剩', '副', '剰', '割', '創', '劇', '劉', '劍', '力', '功', '加', '劣', '助', '努', '劫', '励', '労', '効', '劾', '勁', '勃', '勅', '勇', '勉', '動', '勘', '務', '勝', '募', '勢', '勤', '勧', '勲', '勳', '勺', '勾', '勿', '匁', '匂', '包', '化', '北', '匙', '匠', '匡', '匹', '区', '医', '匿', '十', '千', '升', '午', '半', '卑', '卒', '卓', '協', '南', '単', '博', '卜', '占', '卯', '印', '危', '即', '却', '卵', '卷', '卸', '卿', '厄', '厘', '厚', '原', '厨', '厩', '厳', '去', '参', '又', '叉', '及', '友', '双', '反', '収', '叔', '取', '受', '叙', '叡', '叢', '口', '古', '句', '叩', '只', '叫', '召', '可', '台', '叱', '史', '右', '叶', '号', '司', '各', '合', '吉', '吊', '同', '名', '后', '吏', '吐', '向', '君', '吟', '吠', '否', '含', '吸', '吹', '吻', '吾', '呂', '呈', '呉', '告', '呟', '周', '呪', '味', '呼', '命', '咀', '和', '咲', '咳', '咽', '哀', '品', '哉', '員', '哨', '哩', '哲', '唄', '唆', '唇', '唐', '唯', '唱', '唾', '啄', '商', '問', '啓', '善', '喉', '喋', '喚', '喜', '喝', '喧', '喩', '喪', '喫', '喬', '單', '喰', '営', '嗅', '嗣', '嘆', '嘉', '嘗', '嘘', '嘩', '嘱', '嘲', '噂', '噌', '噛', '器', '噴', '嚇', '嚴', '囚', '四', '回', '因', '団', '困', '囲', '図', '固', '国', '圃', '圈', '國', '圏', '園', '圓', '團', '土', '圧', '在', '圭', '地', '坂', '均', '坊', '坐', '坑', '坦', '坪', '垂', '型', '垢', '垣', '埋', '城', '埜', '域', '埴', '執', '培', '基', '埼', '堀', '堂', '堅', '堆', '堕', '堤', '堪', '堯', '堰', '報', '場', '堵', '堺', '塀', '塁', '塊', '塑', '塔', '塗', '塙', '塚', '塞', '塩', '填', '塾', '境', '墓', '増', '墜', '墨', '墳', '墾', '壁', '壇', '壊', '壌', '壕', '壘', '壞', '士', '壬', '壮', '壯', '声', '壱', '売', '壷', '壽', '変', '夏', '夕', '外', '多', '夜', '夢', '大', '天', '太', '夫', '央', '失', '夷', '奄', '奇', '奈', '奉', '奎', '奏', '契', '奔', '套', '奥', '奧', '奨', '奪', '奬', '奮', '女', '奴', '好', '如', '妃', '妄', '妊', '妖', '妙', '妥', '妨', '妬', '妹', '妻', '姉', '始', '姑', '姓', '委', '姥', '姦', '姪', '姫', '姻', '姿', '威', '娃', '娘', '娠', '娩', '娯', '娼', '婆', '婚', '婦', '婿', '媒', '媛', '嫁', '嫉', '嫌', '嫡', '嬉', '嬢', '孃', '子', '孔', '字', '存', '孜', '孝', '孟', '季', '孤', '学', '孫', '宅', '宇', '守', '安', '宋', '完', '宏', '宕', '宗', '官', '宙', '定', '宛', '宜', '宝', '実', '客', '宣', '室', '宥', '宮', '宰', '害', '宴', '宵', '家', '容', '宿', '寂', '寄', '寅', '密', '富', '寒', '寓', '寛', '寝', '察', '寡', '寢', '實', '寧', '審', '寮', '寵', '寸', '寺', '対', '寿', '封', '専', '射', '将', '將', '專', '尉', '尊', '尋', '導', '小', '少', '尖', '尚', '尤', '尭', '就', '尺', '尻', '尼', '尽', '尾', '尿', '局', '居', '屈', '届', '屋', '屑', '展', '属', '層', '履', '屯', '山', '岐', '岡', '岩', '岬', '岳', '岸', '峠', '峡', '峨', '峯', '峰', '島', '峻', '峽', '崇', '崎', '崖', '崚', '崩', '嵐', '嵩', '嵯', '嶋', '嶺', '巌', '巖', '川', '州', '巡', '巣', '工', '左', '巧', '巨', '差', '己', '已', '巳', '巴', '巷', '巻', '巽', '巾', '市', '布', '帆', '希', '帖', '帝', '帥', '師', '席', '帯', '帰', '帳', '帶', '常', '帽', '幅', '幌', '幕', '幡', '幣', '干', '平', '年', '幸', '幹', '幻', '幼', '幽', '幾', '庁', '広', '庄', '庇', '床', '序', '底', '店', '庚', '府', '度', '座', '庫', '庭', '庵', '庶', '康', '庸', '廃', '廉', '廊', '廟', '廣', '廳', '延', '廷', '建', '廻', '廿', '弁', '弄', '弊', '式', '弐', '弓', '弔', '引', '弘', '弛', '弟', '弥', '弦', '弧', '弱', '張', '強', '弾', '彈', '彌', '当', '彗', '彙', '形', '彦', '彩', '彪', '彫', '彬', '彰', '影', '役', '彼', '往', '征', '径', '待', '律', '後', '徐', '徒', '従', '得', '從', '徠', '御', '復', '循', '微', '徳', '徴', '徹', '徽', '心', '必', '忌', '忍', '志', '忘', '忙', '応', '忠', '快', '念', '忽', '怒', '怖', '怜', '思', '怠', '急', '性', '怨', '怪', '恆', '恋', '恐', '恒', '恕', '恢', '恣', '恥', '恨', '恩', '恭', '息', '恰', '恵', '悉', '悌', '悔', '悟', '悠', '患', '悦', '悩', '悪', '悲', '悼', '情', '惇', '惑', '惚', '惜', '惟', '惠', '惡', '惣', '惧', '惨', '惰', '想', '惹', '惺', '愁', '愉', '意', '愚', '愛', '感', '愼', '慄', '慈', '態', '慌', '慎', '慕', '慢', '慣', '慧', '慨', '慮', '慰', '慶', '憂', '憎', '憐', '憚', '憤', '憧', '憩', '憬', '憲', '憶', '憾', '懇', '應', '懐', '懲', '懷', '懸', '戊', '成', '我', '戒', '或', '戚', '戟', '戦', '戯', '戰', '戲', '戴', '戸', '戻', '房', '所', '扇', '扉', '手', '才', '打', '払', '托', '扱', '扶', '批', '承', '技', '抄', '把', '抑', '投', '抗', '折', '抜', '択', '披', '抱', '抵', '抹', '押', '抽', '拂', '担', '拉', '拍', '拐', '拒', '拓', '拔', '拘', '拙', '招', '拜', '拝', '拠', '拡', '括', '拭', '拳', '拶', '拷', '拾', '持', '指', '按', '挑', '挙', '挟', '挨', '挫', '振', '挺', '挽', '挿', '捉', '捕', '捗', '捜', '捧', '捨', '据', '捲', '捷', '捺', '捻', '掃', '授', '掌', '排', '掘', '掛', '掠', '採', '探', '接', '控', '推', '措', '掬', '掲', '掴', '掻', '揃', '描', '提', '揚', '換', '握', '揮', '援', '揺', '損', '搖', '搜', '搬', '搭', '携', '搾', '摂', '摘', '摩', '摯', '摺', '撃', '撒', '撚', '撞', '撤', '撫', '播', '撮', '撰', '撲', '擁', '操', '擢', '擦', '擬', '攝', '支', '收', '改', '攻', '放', '政', '故', '敍', '敏', '救', '敗', '教', '敢', '散', '敦', '敬', '数', '整', '敵', '敷', '文', '斉', '斎', '斐', '斑', '斗', '料', '斜', '斡', '斤', '斥', '斧', '斬', '断', '斯', '新', '方', '於', '施', '旅', '旋', '族', '旗', '既', '日', '旦', '旧', '旨', '早', '旬', '旭', '旺', '昂', '昆', '昇', '昊', '昌', '明', '昏', '易', '昔', '星', '映', '春', '昧', '昨', '昭', '是', '昴', '昼', '時', '晃', '晄', '晋', '晏', '晒', '晝', '晟', '晦', '晨', '晩', '普', '景', '晴', '晶', '智', '暁', '暇', '暉', '暑', '暖', '暗', '暢', '暦', '暫', '暮', '暴', '曇', '曉', '曖', '曙', '曜', '曝', '曲', '曳', '更', '書', '曹', '曽', '曾', '替', '最', '月', '有', '朋', '服', '朔', '朕', '朗', '望', '朝', '期', '木', '未', '末', '本', '札', '朱', '朴', '机', '朽', '杉', '李', '杏', '材', '村', '杖', '杜', '束', '条', '来', '杭', '杯', '東', '杵', '杷', '松', '板', '枇', '析', '枕', '林', '枚', '果', '枝', '枠', '枢', '枯', '架', '柄', '柊', '柏', '某', '柑', '染', '柔', '柘', '柚', '柱', '柳', '柴', '柵', '査', '柾', '柿', '栃', '栄', '栓', '栖', '栗', '栞', '校', '株', '核', '根', '格', '栽', '桁', '桂', '桃', '案', '桐', '桑', '桔', '桜', '桟', '桧', '桶', '梁', '梅', '梓', '梗', '梛', '條', '梢', '梧', '梨', '梯', '械', '梶', '棄', '棋', '棒', '棚', '棟', '森', '棲', '棺', '椀', '椅', '椋', '植', '椎', '椛', '検', '椰', '椿', '楊', '楓', '楕', '楚', '楠', '楢', '業', '楯', '極', '楼', '楽', '概', '榊', '榎', '榛', '榮', '槇', '構', '槌', '槍', '様', '槙', '槻', '槽', '樂', '樋', '標', '樟', '模', '樣', '権', '横', '樫', '樹', '樺', '樽', '橋', '橘', '橙', '機', '檀', '檎', '檜', '檢', '櫂', '櫓', '櫛', '櫻', '欄', '欠', '次', '欣', '欧', '欲', '欺', '欽', '款', '歌', '歎', '歓', '止', '正', '此', '武', '歩', '歯', '歳', '歴', '死', '殆', '殉', '殊', '残', '殖', '殴', '段', '殺', '殻', '殿', '毅', '母', '毎', '毒', '比', '毘', '毛', '毬', '氏', '民', '気', '氣', '水', '氷', '永', '氾', '汀', '汁', '求', '汎', '汐', '汗', '汚', '汝', '江', '池', '汰', '汲', '決', '汽', '沃', '沈', '沌', '沓', '沖', '沙', '没', '沢', '沫', '河', '沸', '油', '治', '沼', '沿', '況', '泉', '泊', '泌', '法', '泡', '波', '泣', '泥', '注', '泰', '泳', '洋', '洗', '洛', '洞', '津', '洪', '洲', '洵', '洸', '活', '派', '流', '浄', '浅', '浜', '浦', '浩', '浪', '浬', '浮', '浴', '海', '浸', '消', '涙', '涯', '液', '涼', '淀', '淋', '淑', '淡', '淨', '淫', '深', '淳', '淵', '混', '添', '清', '渇', '済', '渉', '渋', '渓', '渚', '減', '渡', '渥', '渦', '温', '測', '港', '湊', '湖', '湘', '湛', '湧', '湯', '湾', '湿', '満', '源', '準', '溜', '溝', '溢', '溶', '溺', '滅', '滉', '滋', '滑', '滝', '滞', '滯', '滴', '漁', '漂', '漆', '漏', '演', '漕', '漠', '漢', '漣', '漫', '漬', '漱', '漸', '潔', '潜', '潟', '潤', '潮', '潰', '澁', '澄', '澪', '澱', '激', '濁', '濃', '濕', '濠', '濡', '濫', '濯', '瀕', '瀧', '瀬', '灘', '火', '灯', '灰', '灸', '灼', '災', '炉', '炊', '炎', '炒', '炭', '点', '為', '烈', '烏', '焚', '無', '焦', '然', '焼', '煉', '煌', '煎', '煙', '煤', '照', '煩', '煮', '熊', '熙', '熟', '熱', '燃', '燈', '燎', '燒', '燕', '燥', '燦', '燭', '燿', '爆', '爪', '爭', '爲', '爵', '父', '爽', '爾', '片', '版', '牌', '牒', '牙', '牛', '牟', '牡', '牧', '物', '牲', '特', '牽', '犀', '犠', '犬', '犯', '状', '狂', '狐', '狗', '狙', '狩', '独', '狭', '狹', '狼', '猛', '猟', '猪', '猫', '献', '猶', '猿', '獄', '獅', '獣', '獲', '獸', '玄', '率', '玉', '王', '玖', '玩', '玲', '珀', '珂', '珈', '珊', '珍', '珠', '班', '現', '球', '理', '琉', '琢', '琥', '琳', '琴', '琵', '琶', '瑚', '瑛', '瑞', '瑠', '瑳', '瑶', '璃', '璧', '環', '璽', '瓜', '瓢', '瓦', '瓶', '甘', '甚', '生', '産', '甥', '用', '甫', '田', '由', '甲', '申', '男', '町', '画', '界', '畏', '畑', '畔', '留', '畜', '畝', '畠', '畢', '略', '番', '異', '畳', '畿', '疊', '疋', '疎', '疏', '疑', '疫', '疲', '疾', '病', '症', '痕', '痘', '痛', '痢', '痩', '痴', '瘍', '療', '癌', '癒', '癖', '発', '登', '白', '百', '的', '皆', '皇', '皐', '皓', '皮', '皿', '盃', '盆', '益', '盗', '盛', '盜', '盟', '盡', '監', '盤', '目', '盲', '直', '相', '盾', '省', '眉', '看', '県', '眞', '真', '眠', '眸', '眺', '眼', '着', '睡', '督', '睦', '睨', '瞥', '瞬', '瞭', '瞳', '矛', '矢', '知', '矩', '短', '矯', '石', '砂', '研', '砕', '砥', '砦', '砧', '砲', '破', '硝', '硫', '硬', '硯', '碁', '碎', '碑', '碓', '碗', '碧', '碩', '確', '磁', '磐', '磨', '磯', '礁', '礎', '示', '礼', '社', '祁', '祇', '祈', '祉', '祐', '祕', '祖', '祝', '神', '祢', '祥', '票', '祭', '祿', '禁', '禄', '禅', '禍', '禎', '福', '禪', '禮', '禰', '禽', '禾', '秀', '私', '秋', '科', '秒', '秘', '租', '秤', '秦', '秩', '称', '移', '稀', '程', '税', '稔', '稚', '稜', '稟', '種', '稲', '稻', '稼', '稽', '稿', '穀', '穂', '積', '穏', '穗', '穣', '穫', '穰', '穴', '究', '空', '穿', '突', '窃', '窄', '窒', '窓', '窟', '窪', '窮', '窯', '窺', '立', '竜', '章', '竣', '童', '竪', '端', '競', '竹', '竺', '竿', '笈', '笑', '笙', '笛', '笠', '符', '第', '笹', '筆', '筈', '等', '筋', '筑', '筒', '答', '策', '箇', '箋', '箔', '箕', '算', '管', '箱', '箸', '節', '範', '篇', '築', '篠', '篤', '簡', '簾', '簿', '籍', '籠', '米', '籾', '粉', '粋', '粒', '粗', '粘', '粛', '粟', '粥', '粧', '粹', '精', '糊', '糖', '糧', '糸', '系', '糾', '紀', '約', '紅', '紋', '納', '紐', '純', '紗', '紘', '紙', '級', '紛', '素', '紡', '索', '紫', '紬', '累', '細', '紳', '紹', '紺', '終', '絃', '組', '絆', '経', '結', '絞', '絡', '絢', '給', '統', '絵', '絶', '絹', '継', '続', '綜', '維', '綱', '網', '綴', '綸', '綺', '綻', '綾', '綿', '緊', '緋', '総', '緑', '緒', '線', '締', '編', '緩', '緯', '練', '緻', '縁', '縄', '縛', '縞', '縣', '縦', '縫', '縮', '縱', '績', '繁', '繊', '繍', '織', '繕', '繭', '繰', '纂', '纏', '纖', '缶', '罪', '置', '罰', '署', '罵', '罷', '羅', '羊', '美', '羚', '羞', '群', '羨', '義', '羽', '翁', '翌', '習', '翔', '翠', '翻', '翼', '耀', '老', '考', '者', '而', '耐', '耕', '耗', '耳', '耶', '耽', '聖', '聘', '聞', '聡', '聴', '職', '聽', '肇', '肉', '肋', '肌', '肖', '肘', '肝', '股', '肢', '肥', '肩', '肪', '肯', '育', '肴', '肺', '胃', '胆', '背', '胎', '胞', '胡', '胤', '胴', '胸', '能', '脂', '脅', '脇', '脈', '脊', '脚', '脩', '脱', '脳', '脹', '腎', '腐', '腔', '腕', '腫', '腰', '腸', '腹', '腺', '膏', '膚', '膜', '膝', '膨', '膳', '臆', '臓', '臟', '臣', '臥', '臨', '自', '臭', '至', '致', '臼', '與', '興', '舌', '舎', '舗', '舜', '舞', '舟', '航', '般', '舵', '舶', '舷', '船', '艇', '艦', '良', '色', '艶', '芋', '芙', '芝', '芥', '芦', '芭', '芯', '花', '芳', '芸', '芹', '芽', '苑', '苔', '苗', '苛', '若', '苦', '英', '苺', '茂', '茄', '茅', '茉', '茎', '茜', '茨', '茶', '茸', '茹', '草', '荒', '荘', '荷', '荻', '莉', '莊', '莞', '莫', '菅', '菊', '菌', '菓', '菖', '菜', '菩', '菫', '華', '菱', '萄', '萌', '萎', '萠', '萩', '萬', '萱', '落', '葉', '著', '葛', '葡', '董', '葦', '葬', '葱', '葵', '葺', '蒐', '蒔', '蒙', '蒲', '蒸', '蒼', '蓄', '蓉', '蓋', '蓑', '蓬', '蓮', '蔑', '蔓', '蔦', '蔭', '蔵', '蔽', '蕃', '蕉', '蕎', '蕗', '蕨', '蕪', '蕾', '薄', '薗', '薙', '薦', '薩', '薪', '薫', '薬', '藁', '藍', '藏', '藝', '藤', '藥', '藩', '藻', '蘇', '蘭', '虎', '虐', '虚', '虜', '虞', '虫', '虹', '虻', '蚊', '蚕', '蛇', '蛋', '蛍', '蛮', '蜂', '蜜', '蝦', '蝶', '融', '螺', '蟹', '血', '衆', '行', '術', '街', '衛', '衝', '衞', '衡', '衣', '表', '衰', '衷', '衿', '袈', '袋', '袖', '被', '袴', '裁', '裂', '装', '裏', '裕', '補', '裝', '裟', '裡', '裳', '裸', '製', '裾', '複', '褐', '褒', '襖', '襟', '襲', '西', '要', '覆', '覇', '見', '規', '視', '覗', '覚', '覧', '親', '観', '覽', '角', '解', '触', '言', '訂', '訃', '計', '訊', '討', '訓', '託', '記', '訟', '訣', '訪', '設', '許', '訳', '訴', '診', '註', '証', '詐', '詔', '評', '詞', '詠', '詢', '詣', '試', '詩', '詫', '詮', '詰', '話', '該', '詳', '誇', '誉', '誌', '認', '誓', '誕', '誘', '語', '誠', '誤', '説', '読', '誰', '課', '誼', '調', '諄', '談', '請', '諏', '諒', '論', '諜', '諦', '諧', '諭', '諮', '諸', '諺', '諾', '謀', '謁', '謂', '謄', '謎', '謙', '講', '謝', '謠', '謡', '謹', '識', '譜', '警', '議', '譲', '護', '讃', '讐', '讓', '谷', '豆', '豊', '豚', '象', '豪', '豹', '貌', '貝', '貞', '負', '財', '貢', '貧', '貨', '販', '貪', '貫', '責', '貯', '貰', '貴', '買', '貸', '費', '貼', '貿', '賀', '賃', '賄', '資', '賊', '賑', '賓', '賛', '賜', '賞', '賠', '賢', '賣', '賦', '質', '賭', '購', '贈', '赤', '赦', '走', '赳', '赴', '起', '超', '越', '趣', '足', '距', '跡', '跨', '路', '跳', '践', '踊', '踏', '踪', '蹄', '蹟', '蹴', '躍', '身', '車', '軌', '軍', '軒', '軟', '転', '軸', '軽', '較', '載', '輔', '輝', '輩', '輪', '輯', '輸', '輿', '轄', '轉', '轍', '轟', '辛', '辞', '辣', '辰', '辱', '農', '辺', '辻', '込', '辿', '迂', '迄', '迅', '迎', '近', '返', '迦', '迪', '迫', '迭', '述', '迷', '追', '退', '送', '逃', '逆', '透', '逐', '逓', '途', '逗', '這', '通', '逝', '逞', '速', '造', '逢', '連', '逮', '週', '進', '逸', '遁', '遂', '遅', '遇', '遊', '運', '遍', '過', '道', '達', '違', '遙', '遜', '遠', '遡', '遣', '遥', '適', '遭', '遮', '遵', '遷', '選', '遺', '遼', '避', '還', '邑', '那', '邦', '邪', '邸', '郁', '郊', '郎', '郡', '部', '郭', '郵', '郷', '都', '鄭', '酉', '酌', '配', '酎', '酒', '酔', '酢', '酪', '酬', '酵', '酷', '酸', '醇', '醉', '醍', '醐', '醒', '醜', '醤', '醸', '釀', '采', '釈', '釉', '里', '重', '野', '量', '金', '釘', '釜', '針', '釣', '釧', '鈍', '鈴', '鉄', '鉛', '鉢', '鉱', '銀', '銃', '銅', '銑', '銘', '銚', '銭', '鋒', '鋭', '鋳', '鋸', '鋼', '錆', '錐', '錘', '錠', '錦', '錫', '錬', '錯', '録', '鍋', '鍛', '鍬', '鍵', '鎌', '鎖', '鎧', '鎭', '鎮', '鏡', '鐘', '鑄', '鑑', '長', '門', '閃', '閉', '開', '閏', '閑', '間', '関', '閣', '閤', '閥', '閲', '闇', '闘', '阜', '阪', '防', '阻', '阿', '陀', '附', '降', '限', '陛', '院', '陣', '除', '陥', '陪', '陰', '陳', '陵', '陶', '陷', '陸', '険', '陽', '隅', '隆', '隈', '隊', '階', '随', '隔', '隙', '際', '障', '隠', '隣', '險', '隷', '隻', '隼', '雀', '雁', '雄', '雅', '集', '雇', '雌', '雑', '雛', '雜', '離', '難', '雨', '雪', '雫', '雰', '雲', '零', '雷', '電', '需', '震', '霊', '霜', '霞', '霧', '露', '青', '靖', '静', '靜', '非', '面', '革', '靴', '鞄', '鞍', '鞘', '鞠', '鞭', '韓', '音', '韻', '響', '頁', '頂', '頃', '項', '順', '須', '頌', '預', '頑', '頒', '頓', '頗', '領', '頬', '頭', '頻', '頼', '題', '額', '顎', '顔', '顕', '願', '顛', '類', '顧', '顯', '風', '颯', '飛', '飜', '食', '飢', '飯', '飲', '飼', '飽', '飾', '餅', '養', '餌', '餓', '館', '饗', '首', '香', '馨', '馬', '馳', '馴', '駄', '駅', '駆', '駈', '駐', '駒', '駕', '駿', '騎', '騒', '験', '騰', '騷', '驍', '驗', '驚', '骨', '骸', '髄', '高', '髪', '髭', '髮', '鬱', '鬼', '魁', '魂', '魅', '魔', '魚', '魯', '鮎', '鮮', '鯉', '鯛', '鯨', '鰯', '鱈', '鱒', '鱗', '鳥', '鳩', '鳳', '鳴', '鳶', '鴨', '鴻', '鵜', '鵬', '鶏', '鶴', '鷄', '鷲', '鷹', '鷺', '鹿', '麒', '麓', '麗', '麟', '麦', '麹', '麺', '麻', '麿', '黄', '黎', '黒', '默', '黙', '黛', '鼎', '鼓', '鼻', '齊', '齢', '龍'];
const predict = async (modelURL) => {
    if (!model) model = await tf.loadModel(modelURL);
    const files = fileInput.files;

    [...files].map(async (img) => {
        const data = new FormData();
        data.append('file', img);
        data.append('total_width', total_width.textContent);
        data.append('total_height', total_height.textContent);
        data.append('width', width.textContent);
        data.append('height', height.textContent);
        data.append('x', x.textContent);
        data.append('y', y.textContent);

        const fetchedJson = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => {
                return result;
            });

        const processedImage = fetchedJson['image'];
        const inputImage = fetchedJson['segment'];
        const JPGImage = fetchedJson['segment_jpg'];
        // shape has to be the same as it was for training of the model
        const prediction = model.predict(tf.reshape(tf.tensor2d(inputImage), shape = [-1, 100, 100, 3]));
        const tmp = prediction.argMax(axis = 1);
        const predictionData = prediction.dataSync();
        var labels = new Array();
        var scores = new Array();
        // console.log(typeof(prediction.argMax(axis = 1)));
        for(var i = 0; i < tmp.size; i++){
            labels.push(tmp.get([i]));
            scores.push(predictionData[i*3095 + labels[i]]);
            console.log(labels[i] + ' ' + scores[i]);
        }
        renderProcessedImg(processedImage);
        renderImageLabel(JPGImage, labels, scores);
    })
};

const renderProcessedImg = (img) => {
    document.getElementById('processedImg').innerHTML = `<img src="${img}">`;
};

const renderImageLabel = (imgs, labels, scores) => {
    result = document.getElementById('result');
    result.innerHTML = ''
    for(var i = 0; i < labels.length; i++){
        // result.innerHTML += `<div class="col-md-2 img-box text-center">
        //                         <div><img src="${imgs[i]}" style="width: 100px"></div>
        //                         <div>${labels[i]}</div>
        //                     </div>`;
        result.innerHTML += `<li style="width: 100px; text-align: center;">
                                <div><img src="${imgs[i]}" style="width: 100%"></div>
                                <div>${labels[i]} - ${str_label[labels[i]]} - ${Math.round(100*scores[i])}%</div>
                            </li>`
    }
};

predictButton.addEventListener("click", () => predict(modelURL));