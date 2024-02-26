<template>
  <div>
  <LoginPage></LoginPage>
</div>
</template>

<script>
import LoginPage from './components/LoginPage.vue';


export default {
  name: 'App',
  components: {
    LoginPage
}
}
</script>

<style>
body{
  background-color: #FDE047;
}
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  background-color: #FDE047;
}
</style>
