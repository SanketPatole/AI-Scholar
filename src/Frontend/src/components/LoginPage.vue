<script>
import { Authenticator } from "@aws-amplify/ui-vue";
import "@aws-amplify/ui-vue/styles.css";
import { Amplify } from "aws-amplify";
import config from "../../src/amplifyconfiguration.json";
import { getCurrentUser } from 'aws-amplify/auth';
import WelcomePage from '@/views/WelcomePage.vue'
import store from "@/store/store";
Amplify.configure(config);
import { Hub } from 'aws-amplify/utils';
Hub.listen('auth', ({ payload }) => {
  switch (payload.event) {
    case 'signedIn':
      localStorage.removeItem('authState');
      console.log(payload);
      localStorage.setItem('signedIn',payload.event);
      location.reload();
      console.log('user have been signedIn successfully.');
      break;
    case 'signedOut':
    localStorage.removeItem('signedIn');
    localStorage.removeItem('authState');
      location.reload();
      console.log('user have been signedOut successfully.');
      break;
    case 'tokenRefresh':
      console.log('auth tokens have been refreshed.');
      break;
    case 'tokenRefresh_failure':
      console.log('failure while refreshing auth tokens.');
      break;
    case 'signInWithRedirect':
      console.log('signInWithRedirect API has successfully been resolved.');
      break;
    case 'signInWithRedirect_failure':
      console.log('failure while trying to resolve signInWithRedirect API.');
      break;
    case 'customOAuthState':
    console.log('custom state returned from CognitoHosted UI');
      break;
  }
});
export default{
  
  components:{
    Authenticator,
    WelcomePage
  },
  data(){
    return{
      chatBotToggle:false,
      currentAuthState:false
    }
  },
  mounted(){
    this.currentAuthenticatedUser();
    console.log(store.state)
  },
  updated(){
    this.currentAuthenticatedUser();
  },
  computed:{
    getAuthState(){
      if(localStorage.getItem('authState')){
        return true;
      }
      else{
        return false;
      }
    }
  },
  methods:{
    async currentAuthenticatedUser() {
  try {
    const { username, userId, signInDetails } = await getCurrentUser();
    console.log(`The username: ${username}`);
    console.log(`The userId: ${userId}`);
    console.log(signInDetails);
  } catch (err) {
    console.log(err);
  }
},
showChatBot(){
  this.chatBotToggle = true
},
getChatBoxToggleStatus(event){
  console.log(event);
  this.chatBotToggle = event;
},
  }
}
</script>


<template>
  <WelcomePage v-if="!getAuthState"  />

  <div v-if="getAuthState" class="pt-12">

  <authenticator>
    <template v-slot="{ user, signOut }">
      <h1>Hello {{ user.signInDetails }}!</h1>
      <button class="btn" @click="signOut">Sign Out</button>
    </template>
  </authenticator>
  </div>

</template>

<style>
.btn{
  width: 110px;
  height: 40px;
  background-color: black;
  color: white;
  border: 0px;
  cursor: pointer;
  border-radius: 20px;
}
</style>