import { fetchAuthSession } from 'aws-amplify/auth'
import { Bedrock } from "@langchain/community/llms/bedrock/web"
import { AmazonKnowledgeBaseRetriever } from "@langchain/community/retrievers/amazon_knowledge_base"
import { ConversationChain, ConversationalRetrievalQAChain } from "langchain/chains"
import { BedrockRuntimeClient, InvokeModelWithResponseStreamCommand } from "@aws-sdk/client-bedrock-runtime"
import { BedrockAgentClient, ListAgentAliasesCommand, ListAgentsCommand, ListKnowledgeBasesCommand } from "@aws-sdk/client-bedrock-agent"
import { BedrockAgentRuntimeClient, RetrieveAndGenerateCommand, RetrieveCommand, InvokeAgentCommand } from "@aws-sdk/client-bedrock-agent-runtime"
import { BedrockClient, ListFoundationModelsCommand } from "@aws-sdk/client-bedrock"


export const getModel = async (modelId = "anthropic.claude-instant-v1") => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]
    const model = new Bedrock({
        model: modelId,
        region: region,
        streaming: true,
        credentials: session.credentials,
        modelKwargs: { max_tokens_to_sample: 1000, temperature: 1 },
    })
    return model
}

export const invokeModelStreaming = async (body, modelId = "anthropic.claude-instant-v1", { callbacks }) => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]
    const client = new BedrockRuntimeClient({ region: region, credentials: session.credentials })
    const input = {
        body: JSON.stringify(body),
        contentType: "application/json",
        accept: "application/json",
        modelId: modelId
    }
    console.log(input)
    const command = new InvokeModelWithResponseStreamCommand(input)
    const response = await client.send(command)

    let decoder = new TextDecoder("utf-8")
    let completion = ""
    for await (const chunk of response.body) {
        const json_chunk = JSON.parse(decoder.decode(chunk.chunk.bytes))
        //console.log(json_chunk)
        let text = ""
        if (json_chunk.type === "content_block_start") text = json_chunk.content_block.text
        if (json_chunk.type === "content_block_delta") text = json_chunk.delta.text
        completion += text
        callbacks?.forEach(callback => {
            if (callback?.handleLLMNewToken) {

                callback.handleLLMNewToken(json_chunk)
            }
        })
        continue

    }
    return completion

}

export const getChain = (llm, memory) => {

    const chain = new ConversationChain({ llm: llm, memory: memory })
    chain.prompt.template = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. 
Current conversation:
{history}

Human: {input}
Assistant:`
    return chain
}


export const getBedrockKnowledgeBases = async () => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]
    const client = new BedrockAgentClient({ region: region, credentials: session.credentials })
    const command = new ListKnowledgeBasesCommand({})
    const response = await client.send(command)
    return response.knowledgeBaseSummaries
}


export const getBedrockAgents = async () => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]

    const client = new BedrockAgentClient({ region: region, credentials: session.credentials })
    const command = new ListAgentsCommand({})
    const response = await client.send(command)

    const agentWithAliases = await Promise.all(response.agentSummaries.map(async agent => {
        const aliases = await getBedrockAgentAliases(client, agent)
        agent.aliases = aliases
        return agent
    }))
    return agentWithAliases
}



export const getBedrockAgentAliases = async (client, agent) => {
    const agentCommand = new ListAgentAliasesCommand({ agentId: agent.agentId })
    const response = await client.send(agentCommand)
    return response.agentAliasSummaries
}



export const ragBedrockKnowledgeBase = async (sessionId, knowledgeBaseId, query, modelId = "anthropic.claude-instant-v1") => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]

    const client = new BedrockAgentRuntimeClient({ region: region, credentials: session.credentials })
    const input = {
        input: { text: query },
        retrieveAndGenerateConfiguration: {
            type: "KNOWLEDGE_BASE",
            knowledgeBaseConfiguration: {
                knowledgeBaseId: knowledgeBaseId,
                modelArn: `arn:aws:bedrock:${region}::foundation-model/${modelId}`
            },
        }
    }

    if (sessionId) {
        input.sessionId = sessionId
    }

    const command = new RetrieveAndGenerateCommand(input)

    try {
        const response = await client.send(command);
        return response;

    } catch (error) {
        return { output: { text: "Error: " + error.message }, citations: [], sessionId }
    }

}

export const invokeBedrockAgent = async (sessionId, agentId, agentAlias, query) => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]

    const client = new BedrockAgentRuntimeClient({ region: region, credentials: session.credentials })
    const input = {
        sessionId: sessionId,
        agentId: agentId,
        agentAliasId: agentAlias,
        inputText: query,
        enableTrace: true,
        //TODO: change inference configuration to a length bigger than 1024 in some settings somewhere
        // added knowledge base configuration to return up to 1 results
        sessionState: {
            knowledgeBaseConfigurations:[{ 
                knowledgeBaseId: "4GPIBKMXWO",
                retrievalConfiguration: { 
                    vectorSearchConfiguration: { 
                        numberOfResults: 20
                    }
                }
            }]
        }
    }

    console.log("input: ", input)

    const command = new InvokeAgentCommand(input)
    const response = await client.send(command,)
    console.log("response:", response)

    let completion = ""
    let references = ""
    let kb = ""

    let decoder = new TextDecoder("utf-8")
    // we know that a response only has 1 chunk: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
    for await (const chunk of response.completion) {
        console.log("chunk:", chunk)

        // ensure that previous data is available to this current process
        if ("chunk" in chunk){ 
            var text = completion + decoder.decode(chunk.chunk.bytes)

            // there are some chunks that don't have attributions
            if ("attribution" in chunk.chunk) {
                const refs = chunk.chunk.attribution.citations
                console.log("refs:", refs)

                let counter = 1
                let seen_citations = {}

                refs.forEach(function(element) {
                    console.log("element:", element)
                    references+= "\n\n\n\n"

                    // find where the reference should be cited in the main text
                    const citeLocation = element.generatedResponsePart.textResponsePart.text
                    const citeInsert = text.split(citeLocation)
                    const citeNumbers = []

                    // for each reference in the list
                    element.retrievedReferences.forEach(function(attr) { 
                        console.log("citation:", attr)

                        // check if citation has been seen before
                        const valuetoFind = attr.content.text
                        const valuesArray = Object.values(seen_citations)
                        const valueExists = valuesArray.includes(valuetoFind)

                        // if the citation is new, add it to object and update references
                        if(!valueExists) {
                            // add citation number to array
                            citeNumbers.push(String(counter))
                            console.log("cite numbers: ", citeNumbers)

                            // update references
                            seen_citations[String(counter)] = valuetoFind
                            references+= "Source: " + String(counter) + "\n\n"
                            references+= "Page Number: " + String(attr.metadata["x-amz-bedrock-kb-document-page-number"]) + "\n\n"
                            references+= "Document: " + attr.metadata["x-amz-bedrock-kb-source-uri"] + "\n\n"
                            references+= valuetoFind
                            references+= "\n\n\n\n\n\n"
                            counter += 1
                            console.log("reference added successfully")
                        }else {
                            // if the citation is not new, find the appropriate number
                            const citationNumber = Object.keys(seen_citations).find(key => seen_citations[key] === valuetoFind)
                            citeNumbers.push(citationNumber)
                            console.log("citeNumbers (with old citation): ", citeNumbers)
                        }

                    // recombine main text of before substring part, substring, sources, source numbers, and remainder of main text
                    text = citeInsert[0] + citeLocation + " Sources: " + citeNumbers.join(", ") + citeInsert[1]
                    });
                });
            }

            // if we just have text, add the text to the completed message
            completion += text
            console.log(text)
        } // else it is a trace chunk
        if ("trace" in chunk){
            if ("trace" in chunk.trace){
                if ("orchestrationTrace" in chunk.trace.trace){
                    if ("observation" in chunk.trace.trace.orchestrationTrace){
                        if ("knowledgeBaseLookupOutput" in chunk.trace.trace.orchestrationTrace.observation){
                            // show outputs of knowledge base query
                            chunk.trace.trace.orchestrationTrace.observation.knowledgeBaseLookupOutput.retrievedReferences.forEach(function(attr) {
                                console.log("kb lookup:", attr)
                                kb+= "Document: " + attr.metadata["x-amz-bedrock-kb-source-uri"] + "\n\n"
                                kb+= "Page Number: " + String(attr.metadata["x-amz-bedrock-kb-document-page-number"]) + "\n\n"
                                kb+= "Text: " +attr.content.text + "\n\n\n---"
                                kb+= "\n\n\n\n"
                            });
                        }
                    }
                }
            }
        }
        // Test prompt: What are some good DACs policies? Please include 10+ relevant sources using a real-time knowledge base query.
    }
    // return the completed message
    return completion + "\n\n\n---End LLM Response---\n\n\n**References:**\n" + references + "\n\n\n**Knowledge Base Query Results:**\n\n\n" + kb 
}


export const retrieveBedrockKnowledgeBase = async (knowledgeBaseId, query) => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]

    const client = new BedrockAgentRuntimeClient({ region: region, credentials: session.credentials })
    const input = { // RetrieveRequest
        knowledgeBaseId: knowledgeBaseId, // required
        retrievalQuery: { // KnowledgeBaseQuery
            text: query, // required
        },
        retrievalConfiguration: { // KnowledgeBaseRetrievalConfiguration
            vectorSearchConfiguration: { // KnowledgeBaseVectorSearchConfiguration
                numberOfResults: 5, // required
            },
        }
    }


    const command = new RetrieveCommand(input)
    const response = await client.send(command)
    return response
}


export const getBedrockKnowledgeBaseRetriever = async (knowledgeBaseId) => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]
    const retriever = new AmazonKnowledgeBaseRetriever({
        topK: 10,
        knowledgeBaseId: knowledgeBaseId,
        region: region,
        clientOptions: { credentials: session.credentials }
    })

    return retriever
}


export const getConversationalRetrievalQAChain = async (llm, retriever, memory) => {


    const chain = ConversationalRetrievalQAChain.fromLLM(
        llm, retriever = retriever)
    chain.memory = memory

    chain.questionGeneratorChain.prompt.template = "Human: " + chain.questionGeneratorChain.prompt.template + "\nAssistant:"

    chain.combineDocumentsChain.llmChain.prompt.template = `Human: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}

Question: {question}
Helpful Answer:
Assistant:`

    return chain
}

/* It's querying the Amazon Bedrock service to fetch a list of available AI models 
from Anthropic that can be used for text generation, based on the provided filters. 
The results can then be used to select which model is most appropriate for the task.  */

export const getFMs = async () => {
    const session = await fetchAuthSession()
    let region = session.identityId.split(":")[0]
    const client = new BedrockClient({ region: region, credentials: session.credentials })
    const input = { byProvider: "Anthropic", byOutputModality: "TEXT",byInferenceType: "ON_DEMAND"}
    const command = new ListFoundationModelsCommand(input)
    const response = await client.send(command)
    return response.modelSummaries
}