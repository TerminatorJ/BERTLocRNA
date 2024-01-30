library(biomaRt)
library(stringi)
library(stringr)
library(purrr)
library(dplyr)
library(httr)
httr::set_config(httr::timeout(600))
##############################################################################################################
#   Calculating the utr and cds sequence from the RNA sequences                                              #
#   Issue: ensemble database may lockdown sometime, and >50% sample will miss according to bioMart matching  #
##############################################################################################################
#TODO
#Solve the NA issue because of the ensemble and refseq sequence mismatches. Some UTR may differ from the refseq transcript sequences.
getsubseq <- function(refseq){
    mart <- useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl", mirror = "asia")
    result1 <- getBM(attributes = c("refseq_mrna",'ensembl_gene_id',
                                'ensembl_transcript_id',
                                 "5utr"),
                 filters = 'refseq_mrna',
                 values = refseq,
                 mart = mart)
    result2 <- getBM(attributes = c("refseq_mrna",
                                 "3utr"),
                 filters = 'refseq_mrna',
                 values = refseq,
                 mart = mart)

    result <- merge(result1, result2, by = "refseq_mrna")
    # print(result)
    result_ddp <- result %>% rename(a5utr = '5utr') %>% rename(a3utr = '3utr') %>% distinct(a5utr, a3utr, .keep_all = TRUE) %>% filter(a5utr != "Sequence unavailable" & a3utr != "Sequence unavailable")
    # print(result_ddp[1:2,c("a5utr", "a3utr")])
    result_ddp <- result_ddp[match(refseq, result_ddp$refseq_mrna), ]
    # print(result_ddp[1:2,c("a5utr", "a3utr")])
    return(result_ddp)

}


# Function to split transcripts based on 5utr and 3utr values
split_transcripts <- function(transcript, a5utr, a3utr, degrees) {
  a5utr.start <- str_locate_all(transcript, a5utr)[[1]][1]
  a5utr.end <- str_locate_all(transcript, a5utr)[[1]][2]
  a3utr.start <- str_locate_all(transcript, a3utr)[[1]][1]
  a3utr.end <- str_locate_all(transcript, a3utr)[[1]][2]

  if (!is.na(a5utr.start) && !is.na(a5utr.end) && !is.na(a3utr.start) && !is.na(a3utr.end)) {
    cds <- substring(transcript, a5utr.end + 1, a3utr.start - 1)
    cds.start <- a5utr.end + 1
    cds.end <- a3utr.start - 1

    return(list('a5utr_start' = a5utr.start, 'a5utr_end' = a5utr.end, 
                'cds_start' = cds.start, 'cds_end' = cds.end, 'cds' = cds,
                'a3utr_start' = a3utr.start, 'a3utr_end' = a3utr.end))
  } else {
    sub_a5utr <- substring(a5utr,ceiling((1-degrees)*nchar(a5utr)),nchar(a5utr))
    sub_a3utr <- substring(a3utr,ceiling((1-degrees)*nchar(a3utr)),nchar(a3utr))
    a5utr.end <- str_locate_all(transcript, sub_a5utr)[[1]][2]
    a3utr.start <- str_locate_all(transcript, sub_a3utr)[[1]][1]
    # print(paste("switch to", a5utr.end, " to ", a3utr.start))
    
    cds <- substring(transcript, a5utr.end + 1, a3utr.start - 1)
    cds.start = a5utr.end + 1
    cds.end = a3utr.start - 1
    # print(cds)
    if (!is.na(cds)) {
      return(list('a5utr_start' = a5utr.start, 'a5utr_end' = a5utr.end, 
                  'cds_start' = cds.start, 'cds_end' = cds.end, 'cds' = cds,
                  'a3utr_start' = a3utr.start, 'a3utr_end' = a3utr.end))
    } else {
      return(list('a5utr_start' = NA, 'a5utr_end' = NA, 
                  'cds_start' = NA, 'cds_end' = NA, 'cds' = NA,
                  'a3utr_start' = NA, 'a3utr_end' = NA))
    }
  }
}

extract_refseq <- function(id) {
      id <- as.character(id)
    refseq_match <- stri_extract(id, regex = "(?<=Refseq_ID:)\\w+")
    if (!is.na(refseq_match)) {
      return(refseq_match)
    } else {
      return(NA)
    }
  }

process <- function(input_file, batch_size, degrees){
  #loading the data
  file1 <- read.csv(file = input_file)
  #extract the refseq from the id columns, and set a new column called "refseq"
  file1 <- file1 %>% mutate(refseq_mrna = sapply(ids, extract_refseq))
  #from refseq_mrna, get the coordinates of the UTRs and CDS regions.
  # print(length(file1$refseq_mrna))
  length <- length(file1$refseq_mrna)
  chunks <- length %/% batch_size + 1
  result_list <- list()
  for (chunk in 1:chunks){
    # print(chunks)
    file2 <- getsubseq(file1$refseq_mrna[1 + (chunk-1)*batch_size: min(chunk*batch_size, length)])
    result_list[[chunk]] <- file2
  }
  # print(result_list[[1]])
  final_result <- do.call(rbind, result_list)

  #merge to files
  merged_file <- merge(file1, final_result, by = "refseq_mrna") %>% distinct(idx, .keep_all = TRUE)


  #add more columns
  merged_file2 <- merged_file  %>% rowwise() %>%
    mutate(
      # Use map to apply the function to each transcript and utr5 and utr3
      a5utr_start = split_transcripts(Xall, a5utr, a3utr, degrees)$a5utr_start,
      a5utr_end = split_transcripts(Xall, a5utr, a3utr, degrees)$a5utr_end,
      cds_start = split_transcripts(Xall, a5utr, a3utr, degrees)$cds_start,
      cds_end = split_transcripts(Xall, a5utr, a3utr, degrees)$cds_end,
      cds = split_transcripts(Xall, a5utr, a3utr, degrees)$cds,
      a3utr_start = split_transcripts(Xall, a5utr, a3utr, degrees)$a3utr_start,
      a3utr_end = split_transcripts(Xall, a5utr, a3utr, degrees)$a3utr_end,
      # Use map_chr to extract the 5utr element from each list
    )  %>% arrange(idx)
    return(merged_file2)
}




###main###
 
# print("train")
# merged_train <- process("/home/sxr280/BERTLocRNA/localization_multiRNA/Train_fold0.csv", batch_size = 500)
# write.csv(merged_train, "../data/Train_fold0_utr_cds.csv", row.names = FALSE)
# print("test")
# merged_test <- process("/home/sxr280/BERTLocRNA/localization_multiRNA/Test_fold0.csv", batch_size = 500)
# write.csv(merged_test, "../data/Test_fold0_utr_cds.csv", row.names = FALSE)
print("validation")
merged_validation <- process("/home/sxr280/BERTLocRNA/localization_multiRNA/Val_fold0.csv", batch_size = 500, degrees = 0.8)
write.csv(merged_validation, "../data/Validation_fold0_utr_cds.csv", row.names = FALSE)
# merged_all <- rbind(c(merged_train, merged_test, merged_validation))
# write.csv(merged_all, "../data/All_fold0_utr_cds.csv", row.names = FALSE)