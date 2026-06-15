<script setup>
import { ref } from 'vue'

const props = defineProps([
  'platform',
  'config',
])

const config = ref(props.config)
</script>

<template>
  <div id="pyrowave-encoder" class="config-page">
    <!-- Chroma Subsampling -->
    <div class="mb-3">
      <label for="pyrowave_chroma" class="form-label">{{ $t('config.pyrowave_chroma') }}</label>
      <select id="pyrowave_chroma" class="form-select" v-model="config.pyrowave_chroma">
        <option value="1">{{ $t('config.pyrowave_chroma_444') }}</option>
        <option value="0">{{ $t('config.pyrowave_chroma_420') }}</option>
      </select>
      <div class="form-text">{{ $t('config.pyrowave_chroma_desc') }}</div>
    </div>
  </div>
</template>
