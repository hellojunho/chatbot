# 토이 챗봇 프로젝트
*한빛 아카데미 - 처음 배우는 딥러닝 챗봇* 에서 진행하는 토이 챗봇 프로젝트 입니다.  


## 프로젝트 디렉토리 구조
- chatbot
    - train_tools : 챗봇 학습 `툴` 관련 파일  
    - models : 챗봇 엔진에서 사용하는 `딥러닝 모델` 관련 파일
        - intent : `의도` 분류 모델 관련 파일
        - ner    : `개체` 인식 모델 관련 파일
    - utils : 챗봇 개발에 필요한 `유틸리티 라이브러리`
    - config : 챗봇 개발에 필요한 `설정`
    - test : 챗봇 개발에 필요한 `테스트 코드`  

## 발생한 에러
1. config 에러  
```commandline
No module named 'config'
```  
이 오류는 내가 프로젝트 디렉토리 구조를 명시해주지 않아서 발생한 에러였다.  
책의 코드를 클론 코딩 하다보면 생길 수 있는 에러로, 자꾸만 저 에러코드가 나오면서 실행이 안된다.  
왜였을까?  
책의 코드는 아래와 같다.  
```commandline
from config.Database import *
```  
하지만, 나는 프로젝트를 조금 더 폴더 단위로 분류하고 싶었던 욕심이 있어서 챗봇 프로젝트를 예제 코드들과 
같은 폴더에 넣었다보니 이런 문제가 생겼다.  
내 정확한 프로젝트의 디렉토리 구조는 아래와 같다.  
`chatbot/chatbot_example_code/chatbot/config, models, test, train_tools, utils`
이래서 나는 코드를 아래와 같이 수정해야했다.  
```commandline
from chatbot.chatbot_example_code.chatbot.config.DatabaseConfig import *
```