'use client'

import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Heading,
  HStack,
  Icon,
  Input,
  Select,
  Spinner,
  Switch,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  Textarea,
  useToast
} from '@chakra-ui/react'
import { FormEventHandler, useCallback, useEffect, useState } from 'react'
import { MdOutlineFileUpload } from 'react-icons/md'
import { LuMousePointer2 } from 'react-icons/lu'
import { IoPlayBack } from 'react-icons/io5'
import { GiGuitarBassHead } from 'react-icons/gi'
import { Link } from '@chakra-ui/next-js'
import { AudioSample, Transcription } from '@/app/types'
import ResizeTextarea from 'react-textarea-autosize'
import CardButton from '@/app/components/CardButton'

export default function Home() {
  const [samples, setSamples] = useState<AudioSample[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string | undefined>()
  const [selectedSample, setSelectedSample] = useState<
    AudioSample | undefined
  >()
  const [transcription, setTranscription] = useState<
    Transcription | undefined
  >()
  const [isChecked, setIsChecked] = useState(false)
  const [isFetchingSamples, setIsFetchingSamples] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [mode, setMode] = useState<String | undefined>()

  const toast = useToast({ position: 'bottom-right' })

  const handleTranscribe: FormEventHandler<HTMLFormElement> = useCallback(
    async (e) => {
      e.preventDefault()
      setIsTranscribing(true)

      if (!selectedSample) {
        setIsTranscribing(false)
        toast({ status: 'error', title: 'No sample selected' })
        return
      }

      if (!selectedModel) {
        toast({ status: 'error', title: 'No model selected' })
        setIsTranscribing(false)
        return
      }

      const res = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: JSON.stringify({
          name: selectedSample.name,
          audio: selectedSample.audio,
          model: selectedModel
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (res.ok) {
        const data = await res.json()
        console.log(data)
        setTranscription(data)
      } else {
        toast({
          status: 'error',
          title: 'Failed to transcribe audio'
        })
      }
      setIsTranscribing(false)
    },
    [selectedSample, selectedModel]
  )

  const handleSelectSample = useCallback(
    (e: any) => {
      const sample = samples.find((s) => s.name === e.target.value)
      if (!sample) {
        return
      }
      setSelectedSample(sample)
    },
    [samples]
  )

  const handleSelectModel = useCallback((e: any) => {
    setSelectedModel(e.target.value)
  }, [])

  const handleFileUpload = useCallback((e: any) => {
    const file = e.target.files[0]
    if (!file) {
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => {
      const audio = e.target?.result
      if (!audio) {
        toast({ status: 'error', title: 'Failed to read audio file' })
        return
      }

      setSelectedSample({
        name: file.name,
        audio: (audio as string).split(',')[1]
      })
    }
    reader.readAsDataURL(file)
  }, [])

  const handleSwitchToggle = useCallback(() => {
    setIsChecked(!isChecked)
  }, [isChecked])

  useEffect(() => {
    fetch('http://localhost:8000/models')
      .then((res) => {
        if (!res.ok) {
          throw new Error('Failed to fetch models')
        }
        return res.json()
      })
      .then((data) => {
        setAvailableModels(data)
      })
      .catch((err) => {
        toast({
          status: 'error',
          title:
            'Failed to fetch models. Either the server is down or there is a network issue.'
        })
        console.warn(err)
      })
  }, [])

  useEffect(() => {
    if (!mode) {
      setSelectedSample(undefined)
      return
    }

    if (mode === 'select' && samples.length === 0) {
      setIsFetchingSamples(true)
      fetch('http://localhost:8000/samples')
        .then((res) => {
          return res.json()
        })
        .then((data) => {
          setSamples(data)
        })
        .catch((err) => {
          console.log(err)
          toast({
            status: 'error',
            title:
              'Failed to fetch samples. Either the server is down or there is a network issue.'
          })
        })
        .finally(() => {
          setIsFetchingSamples(false)
        })
    }
  }, [mode, samples])

  return (
    <Box
      display="flex"
      flexDir="column"
      w="100%"
      h="100vh"
      maxW="1440px"
      mx="auto"
    >
      <HStack position="relative" px={8} py={8} justify="space-between">
        <HStack gap={2} align="center">
          <Icon as={GiGuitarBassHead} boxSize={12} />
          <Heading fontWeight={'normal'}>Automatic Bass Transcription</Heading>
        </HStack>
        <HStack>
          <Link href="https://wp2023.cs.hku.hk/fyp23026/" target="_blank">
            <Button variant="ghost">About</Button>
          </Link>
          <Link href="https://github.com/yawjalik" target="_blank">
            <Button variant="ghost">GitHub</Button>
          </Link>
          <Link href="mailto:yawjalik@gmail.com">
            <Button variant="ghost">Contact</Button>
          </Link>
        </HStack>
      </HStack>
      <HStack
        justify={'space-between'}
        px={8}
        align={'start'}
        flexGrow={1}
        h="100%"
      >
        <form style={{ width: '45%' }} onSubmit={handleTranscribe}>
          <FormControl display="flex" flexDir="column" gap={4} isRequired>
            {!mode && (
              <HStack>
                <CardButton
                  width="50%"
                  text={'Upload a file'}
                  icon={<Icon as={MdOutlineFileUpload} boxSize={10} />}
                  onClick={() => setMode('upload')}
                />
                <CardButton
                  width="50%"
                  text="Select a sample"
                  icon={<Icon as={LuMousePointer2} boxSize={10} />}
                  onClick={() => setMode('select')}
                />
              </HStack>
            )}

            {!!mode && (
              <HStack gap={2}>
                <Icon
                  as={IoPlayBack}
                  boxSize={6}
                  m={0}
                  _hover={{
                    cursor: 'pointer',
                    color: 'primary',
                    transition: 'all 0.1s'
                  }}
                  onClick={() => setMode(undefined)}
                />
                <Text flexGrow={1}>Back</Text>
              </HStack>
            )}

            <HStack width={'100%'} gap={4}>
              {mode === 'upload' && (
                <>
                  <FormLabel
                    htmlFor="upload"
                    border="1px solid lightgray"
                    borderRadius="md"
                    p={2}
                    display="flex"
                    justifyContent="center"
                    alignItems="center"
                    gap={2}
                    width="100%"
                    mx={0}
                    requiredIndicator={false}
                    _hover={{
                      cursor: 'pointer',
                      bg: '#e9e9e9',
                      color: 'black',
                      transition: 'all 0.3s'
                    }}
                  >
                    <Icon as={MdOutlineFileUpload} boxSize={8} />
                    <Text fontWeight="bold">Upload a file</Text>
                  </FormLabel>
                  <Input
                    id="upload"
                    name="upload"
                    type="file"
                    display="none"
                    accept="audio/*"
                    onChange={handleFileUpload}
                  />
                </>
              )}

              {mode === 'select' && isFetchingSamples && (
                <Spinner size="lg" thickness="3px" speed="0.65s" mx={'auto'} />
              )}
              {mode === 'select' && !isFetchingSamples && (
                <Select
                  placeholder={
                    samples.length === 0
                      ? 'No samples available'
                      : 'Select a sample'
                  }
                  onInput={handleSelectSample}
                  flexGrow={1}
                >
                  {samples.map((sample) => (
                    <option key={sample.name} value={sample.name}>
                      {sample.name}
                    </option>
                  ))}
                </Select>
              )}
            </HStack>

            {!!mode && (
              <Select
                placeholder={
                  availableModels.length === 0
                    ? 'No models available'
                    : 'Select a model'
                }
                onInput={handleSelectModel}
              >
                {availableModels.length > 0 &&
                  availableModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
              </Select>
            )}

            {selectedSample && (
              <>
                {selectedSample.lilypond && (
                  <>
                    <HStack align="center">
                      <Switch
                        required={false}
                        isChecked={isChecked}
                        onChange={handleSwitchToggle}
                      />
                      <Text>Show Label</Text>
                    </HStack>
                    {isChecked && (
                      <Textarea
                        as={ResizeTextarea}
                        value={selectedSample.lilypond}
                        isReadOnly
                        fontFamily="mono"
                        transition={'height none'}
                        minRows={10}
                      />
                    )}
                  </>
                )}
                <HStack align="center">
                  <audio
                    controls
                    src={`data:audio/wav;base64,${selectedSample.audio}`}
                  />
                  <Button
                    type="submit"
                    variant="primary"
                    ml="auto"
                    isDisabled={isTranscribing}
                    isLoading={isTranscribing}
                  >
                    Transcribe
                  </Button>
                </HStack>
              </>
            )}
          </FormControl>
        </form>
        <Box w={'50%'} h={'100%'} pb={6}>
          <Box p={2} border="1px solid #e9e9e9" borderRadius="md" h={'100%'}>
            <Tabs
              variant="line"
              colorScheme="purple"
              boxShadow={'sm'}
              h={'100%'}
              display="flex"
              flexDir="column"
            >
              <TabList>
                <Tab>Lilypond</Tab>
                <Tab>PDF</Tab>
                {transcription && !isTranscribing && (
                  <a
                    href={`data:application/pdf;base64,${transcription.pdf}`}
                    download={`${transcription.name}.pdf`}
                    style={{ marginLeft: 'auto', marginBottom: '4px' }}
                  >
                    <Button variant="primary">Download PDF</Button>
                  </a>
                )}
              </TabList>
              <TabPanels fontFamily="mono" flexGrow={1}>
                <TabPanel h={'100%'}>
                  {isTranscribing ? (
                    <Spinner
                      size="xl"
                      speed="0.65s"
                      thickness="4px"
                      color="primary"
                    />
                  ) : (
                    <Textarea
                      as={ResizeTextarea}
                      placeholder={'Your score will appear here'}
                      value={
                        !!transcription ? transcription.lilypond : undefined
                      }
                      color={transcription ? 'cyan.200' : 'lightgray'}
                      padding={0}
                      border={0}
                      minRows={10}
                      transition={'height none'}
                      isReadOnly
                    />
                  )}
                </TabPanel>
                <TabPanel h={'100%'}>
                  {transcription?.pdf ? (
                    <iframe
                      width="100%"
                      height="100%"
                      src={`data:application/pdf;base64,${transcription.pdf}`}
                    />
                  ) : (
                    <Text color="lightgray">Your score will appear here</Text>
                  )}
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        </Box>
      </HStack>
    </Box>
  )
}
