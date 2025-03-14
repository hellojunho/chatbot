# 7장. 챗봇 학습 툴 만들기

## 7.1 MySQL
`MySQL`은 가장 많이 사용하는 오픈소스 `관계형 데이터베이스 관리 시스템 (RDBMS)`이다.  
파이썬을 포함한 다양한 언어에서 사용할 수 있도록 API를 지원하고 있다.  

## 7.2 파이썬으로 데이터베이스 연동하기
파이썬에서 mysql을 이용하려면 mysql 클라이언트 라이브러리를 사용해야 한다.  
하지만 `저수준 API`로 되어있어 직접적으로 사용하기엔 어려울 수도...  
하지만, `고수준 API`를 지원하고 있으며, 무료로 사용이 가능한 `PyMySQL`모듈이 있다!  

아래의 코드로 모듈 불러오기  
```
import pymysql
```  

### 7.2.1 데이터베이스 연결하기
데이터 조작을 위해서는 제일 먼저 MySQL 호스트 DB서버에 연결이 되어있어야 한다.  
```
db = pymysql.connect(
    host='127.0.0.1', 
    user='homestead', 
    passwd='secret', 
    db='homestead', 
    charset='utf8'
)
```  
host: 데이터베잇 서버가 존재하는 호스트 주소  
user: 데이터베이스 로그인 유저  
passwd: 데이터베이스 로그인 패스워드  
db: 데이터베이스 이름  
charset: 데이터베이스에서 사용할 charset 인코딩  

### 7.2.2 데이터 조작하기
데이터 조작 방법은 `DB테이블 생성`, 데이터 `삽입(INSERT)`, `조회(SELCET)`, `변경(UPDATE)`, `삭제(DELETE)` 등이 있다.  

파이썬 코드에 SQL문을 집어넣어 동작시키려면 아래와 같은 방식으로 해보자요.  
```commandline
sql = '''
    CREATE TABLE tb_student (
        id int primary key auto_increment not null,
        name varchar(32),
        age int,
        address varchar(32)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8
    '''
```  

## 7.3 챗봇 학습툴 만들기
학습 데이터를 DB에 저장했을 때 실시간으로 챗봇 시스템에 적용될 수 있도록 제작하는 것이 이번 목표!  
딥러닝 모델의 학습 데이터셋과는 개념이 조금 다르다.  
챗봇 엔진에는 `자연어 처리`를 위한 딥러닝 모델을 사용하는데 이 때 모델 학습을 위해 사용하는 데이터 처리는 8장에서 다룸..  

[챗봇 엔진 입력 처리 과정]  
```commandline
[문장] "내일 오전 10시에 탕수육 주문이 가능할까요?" ->  
[챗봇엔진] ->  
[엔진 해석 결과] 
1. 의도(Intent): 음식주문  
2. 개체명(Named Entity): 내일: Date, 오전 10시: Time, 탕수육: Food  
3. 키워드(Keyword): 내일, 오전, 10시, 탕수육, 주문
```  
두 번째 과정은 엔진에서 해석한 결과를 이용해 학습 DB 내용을 검색한다.  
이 때 해석 결과(의도, 개체명)에 매칭되는 답변 정보다 DB에 존재하면 데이터를 불러와 사용자에게 답변으로 제공.  
[챗봇 엔진 답변 처리 과정]  
```commandline
[챗봇엔진] 답변 검색(의도, 개체명) ->  
[학습 DB] 검색된 답변 데이터 ->  
[답변출력] "주문해주셔서 감사합니다."
```  

### 7.3.1 프로젝트 구조
chatbot
    - train_tools : 챗봇 학습 `툴` 관련 파일  
    - models : 챗봇 엔진에서 사용하는 `딥러닝 모델` 관련 파일
        - intent : `의도` 분류 모델 관련 파일
        - ner    : `개체` 인식 모델 관련 파일
    - utils : 챗봇 개발에 필요한 `유틸리티 라이브러리`
    - config : 챗봇 개발에 필요한 `설정`
    - test : 챗봇 개발에 필요한 `테스트 코드`  
  

`OpenPyXL`모듈을 사용하면 엑셀 파일을 읽어와 DB에 데이터를 저장할 수 있다.  
```commandline
    # 학습 엑셀 파일 불러오기
    wb = openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']
    for now in sheet.iter_rows(min_row=2):
        # 데이터 저장
        insert_data(db, xls_row=)

    wb.close()
```  

#### 7.3.2 당황스러운 오류
여기까지 프로젝트의 구조와 파일들을 작성하면서 오류가 발생했다...  
파이썬의 기초가 다져지지 않았다는 뜻인가..?  
오류를 모르겠다ㅜㅜ  

[파일 경로 오류]  
```commandline
from config.DatabaseConfig import *

```  
여기서 `config.Database`가 오류가 나는데, `config`에만 빨간 밑줄이 쳐진다.  
실행해보면, *No module named 'config'*의 오류 코드가 나옴..  
`pip install config`를 하라고 하지만 해도 안됨..

내 프로젝트 구조는  
```angular2html
chatbot
    - ch01 ~ ch12
    - chatbot
        - config
        - models
        - test
        - train_tools
        - utils
```

해결해버렸다!  
`pip install config`가 아니고! 경로 문제였다...  
내 프로젝트 구조는 `chatbot`이라는 최상위 폴더 아래에 `chatbot`폴더가 또 하나 있었고 이 폴더에 챗봇 프로젝트 코드가 들어있다.  
그러니까! import하려면 `from chatbot.config.DatabaseConfig import *`처럼 해야한다.. ㅎ.ㅎ 파이썬 기초 화이팅!