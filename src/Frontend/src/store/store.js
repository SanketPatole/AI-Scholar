import { createStore } from "vuex";
const store = createStore({
    state () {
      return {
        count: 12,
        authState:false,
        showAuthBox:false
      }
    },
    mutations: {
      increment (state) {
        state.count++
      }
    }
  });

  export default store;