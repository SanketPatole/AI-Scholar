<template>
    <body class="bg-yellow-300">
        <header class="py-4">
            <div class="container  flex justify-between items-center ml-[90px] mt-[10px]">
                <div class="text-white flex">
                    <img src="../assets/scholar.png" alt="" class="w-[140px] h-[70px]">
                </div>
                <nav class="flex space-x-4">
                    <!-- <a href="#" class="text-white bg-black w-[120px] h-[40px] pt-[8px] font-semibold">Chapters</a> -->
                    <!-- <a href="#" class="text-black font-semibold"></a>
                    <a href="#" class="text-black font-semibold">Course</a> -->
                    <div class="card flex justify-content-center" v-if="getSignedInState">
                        <CascadeSelect v-model="selectedCity" @group-change="getCurrentSelectedValueGroup"
                            @change="getCurrentlySelectValue" :options="subjectsArr" optionLabel="cname"
                            optionGroupLabel="name" :optionGroupChildren="['subjects', 'chapters']" style="min-width: 14rem"
                            placeholder="Select a Chapter" />
                    </div>
                    <a href="#" class="text-white bg-black w-[100px] h-[40px] pt-[8px] font-semibold rounded-3xl"
                        v-if="!getSignedInState" @click="openAuth()">Log In</a>
					<a href="#" class="text-white bg-black w-[150px] h-[40px] pt-[8px] font-semibold rounded-3xl"
						v-if="!getSignedInState" @click="enterAsGuest()">Enter as Guest</a>
                    <a href="#" class="text-white bg-black w-[100px] h-[40px] pt-[8px] font-semibold rounded-3xl"
                        v-if="getSignedInState" @click="handleSignOut">Log Out</a>

                </nav>
            </div>
        </header>
        <main class="py-3">
            <div class="container flex ml-[120px]" style="justify-content: space-between;">
                <div>
                    <h1 class=" font-bold text-black mb-4" id="title">
                        <!-- <img src="../assets/scholar.png" alt="" class="w-[150px] h-[70px]"> -->
                    </h1>
                    <p class="text-xl text-black mb-8">
                        Personalized Learning Companion for next Gen Students.</p>
                    <div class="flex bg-black h-[320px] w-[465px] rounded-md" v-if="getSignedInState">
                        <div class=" ml-[33px] bg-white mt-[50px] rounded-md w-[180px] h-[220px]">
                            <div class="card ">
                                <div>
                                    <img src="../assets/img2.png" class="mt-3" alt="">
                                </div>
                                <!-- <div>
                                    <p class="mt-4">Revision Notes</p>
                                </div> -->
                                <Dialog v-model:visible="revisionEmptyModal" header="Empty Field Alert"
                                    :style="{ width: '25rem' }" class="bg-blur">
                                    <div>
                                        Please Select the Class, Subject and Chapter!
                                    </div>
                                </Dialog>
                                <button
                                    class="text-black  bg-white border-[1px] border-black w-[150px] h-[35px] font-light text-[14px] hover:bg-[#FDE047] rounded-3xl mt-[33px]"
                                    @click="executeRevisionNotesFunc"> Create Revision Notes</button>
                                <Dialog v-model:visible="visibleNotesGeneration" modal header="Notes Generation"
                                    :style="{ width: '50rem' }" :breakpoints="{ '1199px': '75vw', '575px': '90vw' }">
                                    <div>
                                        <div v-for="note in revisonNotesArr" :key="note.count" class="m-2">
                                            <span class="font-bold">{{ note.count }}. </span> {{ note.notes }}
                                        </div>
                                    </div>
                                    <div class="w-[100%] flex justify-center" v-show="loadingSpinner">
                                        <img src="../assets/spinner.gif" alt="">
                                    </div>

                                </Dialog>
                            </div>
                        </div>
                        <div class=" ml-[33px] bg-white mt-[50px] rounded-md w-[180px] h-[220px]">
                            <div class="card ">
                                <div>
                                    <img src="../assets/img1.png" class="mt-3" alt="">
                                </div>

                                <button
                                    class="text-black  bg-white border-[1px] border-black w-[150px] h-[35px] font-light text-[14px] hover:bg-[#FDE047] rounded-3xl mt-[25px]"
                                    @click="executeQuestionsAnswersGenerationFunc">Create Practice Test</button>
                                <Dialog v-model:visible="visibleTextGeneration" modal header="Test Generation"
                                    :style="{ width: '50rem' }" :breakpoints="{ '1199px': '75vw', '575px': '90vw' }">
                                    <div>
										<h2 class="font-semibold">A. Fill In the Blanks</h2>
                                        <div v-for=" (qa, index) in questionsAnswersObj['fill_in_the_blanks']" :key="qa" class="mt-6">
											<h3 class="font-semibold">Question {{ index + 1 }}.</h3>
                                            <div v-for="(QAWithOPtions, index) in qa" :key="QAWithOPtions">
                                                {{ QAWithOPtions['fill_in_the_blank_statement'] }}
                                                <ol type="a">
                                                    <li><strong>{{index > 0 ? 'Option '+index+': ' : ''}}</strong>{{ QAWithOPtions['option'] }}</li>
                                                </ol>

                                            </div>
                                        </div>
										<br>
										<h2 class="font-semibold">B. Please answer following questions in 40-50 words.</h2>
                                        <div v-for=" (qa, index) in questionsAnswersObj['question_40_words']" :key="qa" class="mt-6">
                                            <h3 class="font-semibold">Question {{ index + 1 }}.</h3>
                                            <br>
                                            {{ qa['question_40_words'] }}
                                        </div>
										<br>
										<h2 class="font-semibold">C. Please answer following questions in 100-120 words.</h2>
                                        <div v-for=" (qa, index) in questionsAnswersObj['question_100_words']" :key="qa"
                                            class="mt-6">
                                            <h3 class="font-semibold">Question {{ index + 1 }}.</h3>
                                            <br>
                                            {{ qa['question_100_words'] }}
                                        </div>
										<br>
										<h2 class="font-semibold">D. State if True or False.</h2>
                                        <div v-for=" (qa, index) in questionsAnswersObj['true_false']" :key="qa" class="mt-6">
                                            <h3 class="font-semibold">Question {{ index + 1 }}.</h3>
                                            <br>
                                            {{ qa['question'] }}
                                        </div>
                                    </div>
                                    <div class="w-[100%] flex justify-center" v-if="loadingSpinner">
                                        <img src="../assets/spinner.gif" alt="">
                                    </div>

                                </Dialog>
                            </div>
                        </div>

                    </div>
                    <div class="flex bg-black h-[320px] w-[665px] rounded-md" v-if="!getSignedInState">
                        <div class=" ml-[33px] bg-white mt-[50px] rounded-md w-[180px] h-[220px]">
                            <div class="card ">
                                <div>
                                    <img src="../assets/img2.png" class="mt-3" alt="">
                                </div>
                                <div>
                                    <p class="mt-7">Revision Notes</p>
                                </div>
                            </div>
                        </div>
                        <div class=" ml-[33px] bg-white mt-[50px] rounded-md w-[180px] h-[220px]">
                            <div class="card ">
                                <div>
                                    <img src="../assets/img1.png" class="mt-3" alt="">
                                </div>
                                <div>
                                    <p class="mt-5">Practice Test</p>
                                </div>
                            </div>
                        </div>
                        <div class=" ml-[33px] bg-white mt-[50px] rounded-md w-[180px] h-[220px]">
                            <div class="card ">
                                <div>
                                    <img src="../assets/img2.png" class="mt-3" alt="">
                                </div>
                                <div>
                                    <p class="mt-7">Doubt Solving</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mt-10 ml-2 flex justify-center">
                    <img src="../assets/studying.png" class="w-[400px] h-[400px]"
                        alt="Illustration of people engaging in different learning activities, such as reading, using computers, and painting, to represent the variety of courses available.">
                </div>
            </div>
            <div v-if="getSignedInState" @click="showChatBot">
                <ChatBot v-if="currentClass && currentSubject && currentChapter"></ChatBot>
				<div class="giphy-arrow-container" v-if="currentClass && currentSubject && currentChapter">
                    <iframe src="https://giphy.com/embed/UDNftoQb9oYY6x4qoG" width="480" height="480" frameBorder="0" allowFullScreen></iframe>
                </div>
            </div>
            <div class="chatbox-container" v-if="getSignedInState && chatBotToggle">
                <div class="container1">
                    <p class="bot-header text-[20px] mt-[19px]">Scholar Assistant <svg xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20" fill="currentColor" height="25px" width="25px" style="cursor: pointer;"
                            @click="emitEvent">
                            <path
                                d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                        </svg></p>

                    <div class="messageBox mt-8">
                        <template v-for="message in getMessagesArr" :key="message.id">
                            <div class="userMessageContent" v-if="message.from.type === 'user'">
                                {{ message.msg.message }}
                            </div>
                            <div class="chatGptMessageContent" v-if="message.from.type === 'gpt'">
                                {{ message.msg.message }}
                            </div>
                        </template>
                    </div>
                    <div class="inputContainer">
                        <input v-model="currentMessage" type="text" class="messageInput" placeholder="Ask me anything..." />
                        <button @click="sendMessage(currentMessage)" class="askButton">
                            Ask
                        </button>
                    </div>
                </div>
            </div>
            <div>

            </div>
        </main>
        <!-- <footer class="py-6">
            <div class="container mx-auto px-6">
                <p class="text-black text-center">© 2024 MÖNAC. All rights reserved.</p>
            </div>
        </footer> -->
    </body>
</template>

<script>
import { signOut } from 'aws-amplify/auth';
import ChatBot from '@/components/ChatBot.vue';
import chaptersList from '@/config/dictionary_of_chapters.json';
import axios from 'axios';
import Dialog from 'primevue/dialog';

export default {
    components: {
        ChatBot,
        Dialog
    },
    data() {
        return {
            currentMessage: '',
            messages: [],
            getMessagesArr: [],
            chatBotToggle: false,
            visibleTextGeneration: false,
            visibleNotesGeneration: false,
            revisionEmptyModal: false,
            revisonNotesArr: [],
            questionsAnswersObj: {},
            loadingSpinner:false,
            subjectsArr: chaptersList,
            currentClass: '',
            currentSubject: '',
            currentChapter: ''

        }
    },
    computed: {
        getSignedInState() {
            if (localStorage.getItem('signedIn') === 'signedIn') {
                return true;
            }
            else {
                return false;
            }
        }
    },
    methods: {
        openAuth() {
            localStorage.setItem('authState', true);
            location.reload();
        },
		enterAsGuest() {
			localStorage.setItem('signedIn', 'signedIn');
			location.reload();
		},
        getCurrentlySelectValue(event) {
            this.currentChapter = event['value']['cname'];

            console.log('Class: ', this.currentClass);
            console.log('Subject: ', this.currentSubject);
            console.log('Chapter: ', this.currentChapter)

        },
        getCurrentSelectedValueGroup(event) {
            // console.log(Object.keys(event.value))
            if (Object.keys(event.value).includes('subjects')) {
                console.log(event['value'])
                this.currentClass = event['value']['code'];
            }
            if (Object.keys(event.value).includes('chapters')) {
                // console.log(event['value']['name'])
                this.currentSubject = event['value']['name'];
            }
        },
        showChatBot() {
            this.chatBotToggle = true
        },

        async handleSignOut() {
            try {
                await signOut();
                location.reload();
            } catch (error) {
                console.log('error signing out: ', error);
            }
        },
        executeRevisionNotesFunc() {
			this.revisonNotesArr = []
            //console.log('this.revisonNotesArr: ',this.revisonNotesArr)
            //if(this.revisonNotesArr.length > 0){
            //    this.loadingSpinner = false;
            //}
            if (this.currentClass.length > 0 && this.currentSubject.length > 0 && this.currentChapter.length > 0) {
				this.visibleNotesGeneration = true;
                this.loadingSpinner = true;
                let queryParams = `?class=${this.currentClass}&subject=${this.currentSubject}&chapter=${this.currentChapter}`
                axios.get(`https://placeholder_for_notes_generation_api` + queryParams).then((response) => {
                    let count = 1;
                    this.revisonNotesArr = response.data;
                    this.revisonNotesArr.forEach((note) => {
                        note['count'] = count;
                        count++;
                        console.log(note)
                    })
                    this.loadingSpinner = false;
                    console.log(this.revisonNotesArr)
                }).catch((err) => {
                    console.log('Err: ', err)
                })
            }
            else {
                this.revisionEmptyModal = true;
            }


        },
        async sendMessage(message) {
            if (message.length > 0) {
                    let msgArr1 = {
                        "from": {
                            "type": "user"
                        },
                        "msg": {
                            "message": message

                        }
                    }
                    this.getMessagesArr.push(msgArr1);
                let queryParams = `?class=${this.currentClass}&subject=${this.currentSubject}&chapter=${this.currentChapter}&message=${message}`
                await axios.get(`https://placeholder_for_doubt_solving_api` + queryParams).then((response) => {
                    console.log(response.data);
                    let msgArr2 = {
                        "from": {
                            "type": "gpt"
                        },
                        "msg": {
                            "message": response.data.msg.message

                        }
                    }
                    this.getMessagesArr.push(msgArr2);
                    this.currentMessage = '';
                }).catch((err) => {
                    console.log('Caught an Error: ', err)
                })

            }
        },
        emitEvent() {
            this.chatBotToggle = false
        },
        executeQuestionsAnswersGenerationFunc() {
			this.questionsAnswersObj = []
            //if(this.questionsAnswersObj.length > 0){
            //    this.loadingSpinner = false;
            //}
            if (this.currentClass.length > 0 && this.currentSubject.length > 0 && this.currentChapter.length > 0) {
				this.visibleTextGeneration = true
                 this.loadingSpinner = true;
                let queryParams = `?class=${this.currentClass}&subject=${this.currentSubject}&chapter=${this.currentChapter}`
                axios.get(`https://placeholder_for_test_generation_api` + queryParams).then((response) => {
                    this.questionsAnswersObj = response.data;
                    console.log(this.questionsAnswersObj);
                    this.loadingSpinner = false;

                }).catch((error) => {
                    console.log(error)
                })
            }
            else {
                this.revisionEmptyModal = true;
            }

        }


    }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Cedarville+Cursive&display=swap');

.p-cascadeselect .p-cascadeselect-label.p-placeholder {
    color: red !important;
}

body {
  transform: scale(0.90);
  transform-origin: top left;
  width: 100%;
  height: 100%;
}




.chatbox-container {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 1000;
}

.giphy-arrow-container {
  position: fixed;
  bottom: 24px;
  right: 100px;
  z-index: 1001;
  pointer-events: none;
}

.giphy-arrow-container iframe {
  width: 100px;
  height: 100px;
}

.p-cascadeselect.p-component.p-inputwrapper {
    min-width: 14rem;
    background: black;
    border-radius: 29px;
    height: 45px;
}

.p-cascadeselect .p-cascadeselect-label.p-placeholder {
    color: white !important;
}

@font-face {
    font-family: 'google-cursive';
    src: url('https://fonts.googleapis.com/css2?family=Cedarville+Cursive&display=swap');
}

#title {
    font-family: 'google-cursive';
}

.container1 {
    width: 300px;
    height: 500px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-family: 'Roboto', sans-serif;
}

h1 {
    font-size: 24px;
    font-weight: 500;
    text-align: center;
    color: #222;
    padding: 16px;
    margin: 0;
}

.messageBox {
    padding: 16px;
    flex-grow: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.messageFromUser,
.messageFromChatGpt {
    display: flex;
}



.messageBox {
    max-height: 400px;
    overflow-y: auto;
    padding: 0 16px;
    border-top: 1px solid #f0f0f0;
    border-bottom: 1px solid #f0f0f0;
    flex-grow: 1;
}

.messageFromUser,
.messageFromChatGpt {
    display: flex;
    margin-bottom: 8px;
}

.userMessageWrapper,
.chatGptMessageWrapper {
    display: flex;
    flex-direction: column;
}

.userMessageWrapper {
    align-self: flex-end;
}

.chatGptMessageWrapper {
    align-self: flex-start;
}

.userMessageContent,
.chatGptMessageContent {
    max-width: 70%;
    padding: 8px 12px;
    border-radius: 18px;
    margin-bottom: 2px;
    font-size: 14px;
    line-height: 1.4;
}

.userMessageContent {
    background-color: #1877F2;
    color: white;
    border-top-left-radius: 0;
    margin-right: 30px;
}

.chatGptMessageContent {
    background-color: #EDEDED;
    color: #222;
    border-top-right-radius: 0;
    margin-left: 30%;
}

.userMessageTimestamp,
.chatGptMessageTimestamp {
    font-size: 10px;
    color: #999;
    margin-top: 2px;
}

.userMessageTimestamp {
    align-self: flex-end;
}

.chatGptMessageTimestamp {
    align-self: flex-start;
}

.inputContainer {
    display: flex;
    align-items: center;
    padding: 8px;
    background-color: #f0f0f0;
}

.messageInput {
    flex-grow: 1;
    border: none;
    outline: none;
    padding: 12px;
    font-size: 16px;
    background-color: white;
    border-radius: 24px;
    margin-right: 8px;
}

.askButton {
    background-color: #1877F2;
    color: white;
    font-size: 16px;
    padding: 8px 16px;
    border: none;
    outline: none;
    cursor: pointer;
    border-radius: 24px;
    transition: background-color 0.3s ease-in-out;
}

.askButton:hover {
    background-color: #145CB3;
}

.bot-header {
    display: flex;
    justify-content: space-evenly;
}

@media (max-width: 480px) {
    .container {
        width: 100%;
        max-width: none;
        border-radius: 0;
    }
}

.chatbox-container {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 1000;
}


.messageBox {
    padding: 16px;
    flex-grow: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.messageFromUser,
.messageFromChatGpt {
    display: flex;
}</style>